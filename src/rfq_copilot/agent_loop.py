"""v1: a hand-written tool-calling loop. The model decides the order of the
steps; code decides what actually happened (the action) from the tool results.

    conv = Conversation()
    run = conv.send("Price me a call on SX5E")      # -> clarify
    run = conv.send("3m 105%, 10mm EUR")            # -> price, pending approval

`run` follows the contract in attack_checks.py.
"""
import time

from rfq_copilot import llm
from rfq_copilot.agent_tools import TOOL_DECLARATIONS, TOOLS, rfq_from_args
from rfq_copilot.prompts import SYSTEM_PROMPT

MAX_STEPS = 6           # LLM calls per client message: validate, price, reply + room for retries
MAX_CLARIFICATIONS = 2  # after this many questions in a row, stop asking and hand over to a human

# Used whenever code overrides the model's outcome: the model's own text was written
# for a different outcome (a question, or even a quote with numbers that were never priced).
HANDOFF_REPLY = "Thanks for your request. We can't quote this automatically; a salesperson will follow up shortly."


def _wrap(client_text: str) -> str:
    # Remove the closing tag so the client can't "end" the data block and
    # write text that looks like it comes from outside it.
    return f"<client_message>{client_text.replace('</client_message>', '')}</client_message>"


class Conversation:
    def __init__(self, chat=llm.chat):
        self.chat = chat            # injectable: tests pass a scripted fake
        self.messages: list[dict] = []
        self.clarifications = 0

    def send(self, client_text: str) -> dict:
        """Handle one client message: run the loop, then derive the outcome."""
        start = time.perf_counter()
        self.messages.append({"role": "user", "text": _wrap(client_text)})
        trace: list[dict] = []      # every tool call this turn, outer and inner
        tokens = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
        reply, steps = None, 0

        for steps in range(1, MAX_STEPS + 1):
            resp = self.chat(SYSTEM_PROMPT, self.messages, TOOL_DECLARATIONS)
            for k in tokens:
                tokens[k] += resp.usage.get(k, 0)
            # The model's own turn goes in first: tool results must answer it.
            self.messages.append(resp.as_message())

            if not resp.tool_calls:
                reply = resp.text
                break

            results = [{"id": call.id, "name": call.name, "result": self._run_tool(call, trace)}
                       for call in resp.tool_calls]
            # One message answering every call of that turn.
            self.messages.append(llm.tool_results_message(results))

        return self._outcome(trace, reply, steps, tokens, time.perf_counter() - start)

    def _run_tool(self, call, trace: list[dict]) -> dict:
        entry = {"name": call.name, "args": call.args, "result": None}
        trace.append(entry)  # appended before running, so inner calls are logged after their parent
        fn = TOOLS.get(call.name)
        if fn is None:
            entry["result"] = {"error": f"unknown tool {call.name!r}"}
        else:
            try:
                entry["result"] = fn(call.args, trace)
            except Exception as e:
                # Bad arguments, a failed guard, or a tool bug: the model sees the error
                # and can correct itself; the run never crashes on a tool.
                entry["result"] = {"error": f"{type(e).__name__}: {e}"}
        return entry["result"]

    def _outcome(self, trace, reply, steps, tokens, latency) -> dict:
        """The action comes from tool results, never from what the model says."""
        validations = [c for c in trace if c["name"] == "validate_rfq" and "error" not in c["result"]]
        prices = [c for c in trace if c["name"] == "price_rfq" and "error" not in c["result"]]
        last = validations[-1] if validations else None

        quote, rfq, overridden = None, None, False
        if prices:
            action, reasons = "price", []
            quote, rfq = prices[-1]["result"], prices[-1]["args"]
        elif last is None:
            action, reasons, overridden = "reject", ["no_valid_parse"], True
        else:
            action, reasons = last["result"]["action"], last["result"]["reasons"]
            rfq = last["args"]
            if action == "price":
                # Validated but never priced
                action, reasons, overridden = "reject", ["not_priced"], True
        if reply is None:
            action, reasons, overridden = "reject", ["step_budget_exceeded"], True

        if action == "clarify" and self.clarifications >= MAX_CLARIFICATIONS:
            action, reasons, overridden = "reject", ["max_clarifications_reached"], True
        # Counts questions in a row; any other outcome starts afresh.
        self.clarifications = self.clarifications + 1 if action == "clarify" else 0
        if overridden:
            reply, quote = HANDOFF_REPLY, None

        return {
            "action": action,
            "reasons": reasons,
            "reply": reply or "",
            "rfq": rfq_from_args(rfq).model_dump() if rfq is not None else None,
            "quote": quote,
            "tool_calls": trace,
            "approval_requested": action == "price",  # a priced draft always waits for a human
            "sent": False,                            # nothing in this loop can send
            "steps": steps,
            "latency_s": latency,
            "tokens": tokens,
        }


def run_agent(client_text: str, chat=llm.chat) -> dict:
    """One message, fresh conversation."""
    return Conversation(chat).send(client_text)
