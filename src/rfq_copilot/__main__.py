"""Interactive demo

    python -m rfq_copilot          # v1, the hand-written loop
    python -m rfq_copilot --v2     # v2, the LangGraph graph: approval resumes its interrupt

Each line is one client message in the same conversation. A priced draft waits
for approval; nothing is ever sent (the agent has no send tool), the
decision is only printed.
"""
import json
import os
import sys

from rfq_copilot import llm
from rfq_copilot.agent_loop import Conversation
from rfq_copilot.graph import GraphConversation

REQUIRED_ENV = ("LLM_MODEL", "GEMINI_KEY")
HELP = "Type a client message. /new starts a new conversation, /quit or Ctrl-D exits."


def show(run: dict) -> None:
    print(f"\n[{run['action']}] {', '.join(run['reasons']) or 'ok'}  ({run['steps']} steps, {run['latency_s']:.1f}s)")
    if run["rfq"]:
        print("rfq:  ", json.dumps({k: v for k, v in run["rfq"].items() if v not in (None, [])}))
    if run["quote"]:
        q = run["quote"]
        print(f"quote: {q['currency']} {q['premium']:,.0f} ({q['premium_pct_notional']:.2f}% of notional), "
              f"K={q['strike_abs']:.2f}, T={q['T_years']:.3f}y, spot={q['spot']}, vol={q['vol']:.1%}")
    print(f"reply: {run['reply']}\n")


def repl(chat=llm.chat, agent=Conversation) -> None:
    print(HELP)
    conv = agent(chat)
    while True:
        try:
            text = input("client> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text in ("/quit", "/exit"):
            return
        if text == "/new":
            conv = agent(chat)
            print("(new conversation)")
            continue
        if not text:
            continue

        try:
            run = conv.send(text)
        except llm.QuotaExhausted as e:
            print(f"error: {e}", file=sys.stderr)
            continue
        show(run)
        if run["approval_requested"]:
            ok = input("trader: approve this quote? [y/N] ").strip().lower() == "y"
            if isinstance(conv, GraphConversation):
                # v2: the decision resumes the paused graph, which records it in its state
                sent = conv.decide("approve" if ok else "reject")["sent"]
                print(f"graph resumed, sent={sent} (demo: nothing leaves this machine)\n")
            else:
                print("approved (demo: not sent)\n" if ok else "rejected, nothing sent\n")


def main() -> int:
    missing = [v for v in REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        print(f"missing env vars: {', '.join(missing)}. Did you `conda activate rfqco`?", file=sys.stderr)
        return 1
    repl(agent=GraphConversation if "--v2" in sys.argv[1:] else Conversation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
