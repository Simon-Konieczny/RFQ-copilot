"""v2: the same agent as an explicit LangGraph state graph. Code fixes the order of
the steps; the LLM only does the language: reading the client (parse) and writing
a clarifying question (clarify). The quote text is a template filled in code.

    parse -> validate -> price -> draft -> approval (interrupt: waits for a trader)
                     |-> clarify
                     |-> reject

    conv = GraphConversation()
    run = conv.send("Price me a call on SX5E")      # -> clarify
    run = conv.send("3m 105%, 10mm EUR")            # -> price, paused at approval
    state = conv.decide("approve")                  # the trader resumes the graph

`run` follows the same contract as v1 (attack_checks.py), so the eval scores both alike.
"""
import json
import operator
import time
import uuid
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from rfq_copilot import llm
from rfq_copilot.agent_loop import HANDOFF_REPLY, MAX_CLARIFICATIONS, _wrap
from rfq_copilot.agent_tools import RFQ_PARAMETERS, rfq_from_args
from rfq_copilot.prompts import CLARIFY_PROMPT, PARSE_PROMPT
from rfq_copilot.tools import price_rfq, validate_rfq

# The only tool the model sees in v2: it can describe the RFQ, nothing else.
PARSE_TOOL = {"name": "submit_rfq",
              "description": "Submit the RFQ exactly as the client stated it.",
              "parameters": RFQ_PARAMETERS}
TOKEN_KEYS = ("input_tokens", "output_tokens", "thinking_tokens")


class State(TypedDict, total=False):
    # Kept across turns: the checkpointer stores the state per conversation thread.
    client_messages: Annotated[list[str], operator.add]  # reducer: each turn appends, never overwrites
    clarifications: int     # questions asked in a row
    # Per turn: parse resets them, so nothing from the previous turn leaks into this one.
    args: dict | None       # submit_rfq arguments: the RFQ as the client stated it
    rfq: dict | None        # args as a validated RFQModel dump
    action: str             # price | clarify | reject
    reasons: list[str]
    quote: dict | None      # price_rfq result: every number comes from here
    reply: str
    trace: list[dict]       # every tool call, same shape as v1
    llm_calls: int
    tokens: dict
    sent: bool


def draft_quote(args: dict, q: dict) -> str:
    """The quote text, filled from price_rfq's numbers: no LLM between pricer and client."""
    unit = args["strike_unit"]
    strike = unit.upper() if unit in ("atm", "atmf") else f"{args['strike_value']:g}{'%' if unit == 'pct_spot' else ''}"
    return (f"{q['underlying']} {args['expiry_value'].upper()} {strike} {q['option_type']}, "
            f"{q['currency']} {q['notional']:,.0f}: {q['premium_pct_notional']:.2f}% of notional "
            f"({q['currency']} {q['premium']:,.0f}), spot {q['spot']:g}, strike {q['strike_abs']:.2f}, "
            f"vol {q['vol']:.1%}. Indicative, subject to trader approval.")


def _plain(x):
    """numpy scalars -> Python numbers, recursively. scipy's norm.cdf makes price_option return
    numpy.float64, which the checkpointer can't serialise (same .item() idiom as get_market_data)."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    return x.item() if hasattr(x, "item") else x


def _add_usage(state: State, resp: llm.LLMResponse) -> dict:
    return {"llm_calls": state.get("llm_calls", 0) + 1,
            "tokens": {k: state.get("tokens", {}).get(k, 0) + resp.usage.get(k, 0) for k in TOKEN_KEYS}}


def build_graph(chat=llm.chat, checkpointer=None):
    """chat is injectable (tests pass a scripted fake), like v1's Conversation."""

    def parse(state: State) -> dict:
        # The whole client side of the conversation, re-read every turn: the RFQ is a
        # function of everything the client said, so there is no merge step to get wrong.
        text = "\n".join(_wrap(m) for m in state["client_messages"])
        resp = chat(PARSE_PROMPT, [{"role": "user", "text": text}], [PARSE_TOOL], force_tool=PARSE_TOOL["name"])
        calls = [c for c in resp.tool_calls if c.name == PARSE_TOOL["name"]]
        fresh = {"rfq": None, "action": "", "reasons": [], "quote": None, "reply": "", "trace": [],
                 "sent": False, "llm_calls": 0, "tokens": {}}
        return {**fresh, **_add_usage(fresh, resp), "args": calls[0].args if calls else None}

    def validate(state: State) -> dict:
        trace = list(state["trace"])
        if state["args"] is None:
            action, reasons, rfq = "reject", ["no_valid_parse"], None
        else:
            entry = {"name": "validate_rfq", "args": state["args"], "result": None}
            trace.append(entry)
            try:
                model = rfq_from_args(state["args"])
                entry["result"] = validate_rfq(model)
                action, reasons, rfq = entry["result"]["action"], entry["result"]["reasons"], model.model_dump()
            except ValueError as e:
                # v1 hands the error back to the model to retry; here a human takes over.
                entry["result"] = {"error": f"{type(e).__name__}: {e}"}
                action, reasons, rfq = "reject", ["no_valid_parse"], None

        if action == "clarify" and state.get("clarifications", 0) >= MAX_CLARIFICATIONS:
            action, reasons = "reject", ["max_clarifications_reached"]
        clarifications = state.get("clarifications", 0) + 1 if action == "clarify" else 0
        return {"trace": trace, "action": action, "reasons": reasons, "rfq": rfq, "clarifications": clarifications}

    def price(state: State) -> dict:
        trace = list(state["trace"])
        entry = {"name": "price_rfq", "args": state["args"], "result": None}
        trace.append(entry)  # before running, so the inner calls price_rfq logs come after it (as in v1)
        try:
            entry["result"] = price_rfq(rfq_from_args(state["args"]), trace)
        except ValueError as e:
            entry["result"] = {"error": f"{type(e).__name__}: {e}"}
            return {"trace": _plain(trace), "action": "reject", "reasons": ["not_priced"]}
        return {"trace": _plain(trace), "quote": _plain(entry["result"])}

    def draft(state: State) -> dict:
        return {"reply": draft_quote(state["args"], state["quote"])}

    def approval(state: State) -> dict:
        # interrupt() pauses the graph here and the checkpointer keeps the state until a
        # trader resumes it with Command(resume="approve" | "reject"). No node before
        # this one can send, and no tool lets the model skip it.
        decision = interrupt({"reply": state["reply"], "quote": state["quote"]})
        return {"sent": decision == "approve"}

    def clarify(state: State) -> dict:
        # The writer sees reasons and parsed fields only, never the client's text.
        understood = {k: v for k, v in state["args"].items()
                      if k not in ("request_type", "unsupported_reason", "ambiguities")}
        payload = json.dumps({"understood": understood, "ask_about": state["reasons"]})
        resp = chat(CLARIFY_PROMPT, [{"role": "user", "text": payload}], None)
        fallback = f"Could you give us: {', '.join(r.replace('_', ' ') for r in state['reasons'])}?"
        return {"reply": resp.text.strip() or fallback, **_add_usage(state, resp)}

    def reject(state: State) -> dict:
        return {"reply": HANDOFF_REPLY, "quote": None}

    g = StateGraph(State)
    for node in (parse, validate, price, draft, approval, clarify, reject):
        g.add_node(node.__name__, node)
    g.add_edge(START, "parse")
    g.add_edge("parse", "validate")
    # Code routes on validate_rfq's action: the model never picks the next step.
    g.add_conditional_edges("validate", lambda s: s["action"], ["price", "clarify", "reject"])
    g.add_conditional_edges("price", lambda s: "draft" if s["quote"] else "reject", ["draft", "reject"])
    g.add_edge("draft", "approval")
    for node in ("approval", "clarify", "reject"):
        g.add_edge(node, END)
    return g.compile(checkpointer=checkpointer)


class GraphConversation:
    """Same interface as v1's Conversation: send(client_text) -> run dict."""

    def __init__(self, chat=llm.chat):
        # The checkpointer is what makes interrupt/resume and multi-turn state work.
        self.graph = build_graph(chat, InMemorySaver())
        self.config = {"configurable": {"thread_id": uuid.uuid4().hex}}

    def send(self, client_text: str) -> dict:
        start = time.perf_counter()
        state = self.graph.invoke({"client_messages": [client_text]}, self.config)
        return _run(state, time.perf_counter() - start)

    def decide(self, decision: str) -> dict:
        """The trader's answer to a paused quote: "approve" or "reject". Returns the final state."""
        return self.graph.invoke(Command(resume=decision), self.config)


def _run(state: dict, latency: float) -> dict:
    return {
        "action": state["action"],
        "reasons": state["reasons"],
        "reply": state["reply"],
        "rfq": state["rfq"],
        "quote": state["quote"],
        "tool_calls": state["trace"],
        "approval_requested": "__interrupt__" in state,  # the graph stopped at the approval node
        "sent": state["sent"],
        "steps": state["llm_calls"],  # LLM calls, as v1's steps
        "latency_s": latency,
        "tokens": state["tokens"],
    }


def run_graph(client_text: str, chat=llm.chat) -> dict:
    """One message, fresh conversation."""
    return GraphConversation(chat).send(client_text)


def mermaid() -> str:
    """The graph as a mermaid diagram (for the README)."""
    return build_graph().get_graph().draw_mermaid()
