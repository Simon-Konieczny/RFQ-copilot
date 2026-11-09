"""Score agents on the eval set and compare them side by side.

    python eval/run_eval.py                          # v1 and v2, every row + dialogues
    python eval/run_eval.py --agents v2 --split dev --limit 10

Writes eval/results.md (comparison + failure report), eval/report.html (the same,
clickable: every row with both agents' replies and traces; `make report` opens it)
and eval/runs/<agent>.jsonl (raw runs with their traces). LLM responses are cached in
.cache/llm, so re-running unchanged prompts costs no quota; if the quota runs out
mid-run, the rows done so far are cached and a re-run picks up from there for free.
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

from report import write_html

from rfq_copilot import llm
from rfq_copilot.agent_loop import Conversation
from rfq_copilot.attack_checks import resisted
from rfq_copilot.graph import GraphConversation
from rfq_copilot.schema import RFQModel
from rfq_copilot.tools import get_market_data

ROOT = Path(__file__).resolve().parents[1]
AGENTS = {"v1": Conversation, "v2": GraphConversation}
FIELDS = list(RFQModel.model_fields)
PRICE_TOL = 1e-5  # labels are rounded to 6 decimals

# What a run that crashed looks like, so it scores as a failure instead of stopping the eval.
ERROR_RUN = {"action": "error", "reasons": [], "reply": "", "rfq": None, "quote": None, "tool_calls": [],
             "approval_requested": False, "sent": False, "steps": 0, "latency_s": 0.0, "tokens": {}}


class Meter:
    """Wraps llm.chat to count calls and cache hits: a cached call takes ~0 s, so
    latency is only meaningful for runs where every call was live."""

    def __init__(self):
        self.calls = self.cached = 0

    def __call__(self, *args, **kwargs):
        resp = llm.chat(*args, **kwargs)
        self.calls += 1
        self.cached += resp.cached
        return resp


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------- scoring one run ----------

def normalise(rfq: dict | None) -> dict:
    """Put a predicted or gold RFQ in the labels' conventions before comparing:
    canonical ticker, currency filled from the underlying when not stated, sorted flags."""
    if rfq is None:
        return {}
    r = dict(rfq)
    mkt = get_market_data(r["underlying"]) if r.get("underlying") else None
    if mkt:
        r["underlying"] = mkt["ticker"]
        r["currency"] = r.get("currency") or mkt["currency"]
    elif r.get("underlying"):
        r["underlying"] = r["underlying"].strip().upper()
    if r.get("currency"):
        r["currency"] = r["currency"].strip().upper()
    if r.get("expiry") and r["expiry"].get("value"):
        r["expiry"] = {**r["expiry"], "value": r["expiry"]["value"].strip().upper()}
    r["ambiguities"] = sorted(r.get("ambiguities") or [])
    return r


def price_ok(run: dict, expected: float) -> bool:
    q = run.get("quote")
    return bool(q) and abs(q["price_per_unit"] - expected) <= PRICE_TOL


def score_row(row: dict, run: dict) -> dict:
    s = {"id": row["id"], "category": row["category"], "expected": row["expected_action"],
         "action": run["action"], "action_ok": run["action"] == row["expected_action"],
         "reasons_ok": sorted(run["reasons"]) == sorted(row["reasons"])}
    # Parsing is scored only where there is an RFQ to parse; chat and unsupported
    # requests are judged on the action and reasons above.
    if row["gold_rfq"]["request_type"] == "rfq":
        pred, gold = normalise(run["rfq"]), normalise(row["gold_rfq"])
        s["field_ok"] = {f: pred.get(f) == gold[f] for f in FIELDS}
        s["field_diff"] = {f: [pred.get(f), gold[f]] for f, ok in s["field_ok"].items() if not ok}  # [got, gold]
    if row["expected_action"] == "price":
        s["price_ok"] = price_ok(run, row["expected_pricing"]["price_per_unit"])
    if row["attack"]:
        # Handled = the right action AND every must_not check passes on the trace.
        s["attack_ok"] = s["action_ok"] and resisted(row, run)
    return s


# ---------- running ----------

def run_one(agent: str, messages: list[str]) -> tuple[list[dict], Meter]:
    """Send messages in one conversation; one run per message."""
    meter = Meter()
    conv = AGENTS[agent](meter)
    runs = []
    for m in messages:
        try:
            runs.append(conv.send(m))
        except llm.QuotaExhausted:
            raise
        except Exception as e:  # one broken row must not sink the whole eval
            runs.append({**ERROR_RUN, "reasons": [f"{type(e).__name__}: {e}"]})
    return runs, meter


def evaluate(agent: str, rows: list[dict], dialogues: list[dict]) -> dict:
    out = {"rows": [], "dialogues": [], "stopped": None}
    try:
        for row in rows:
            (run,), meter = run_one(agent, [row["message"]])
            out["rows"].append({"row": row, "score": score_row(row, run), "run": run,
                                "calls": meter.calls, "cached": meter.cached})
            print(f"  {agent} {row['id']:8} {'ok  ' if out['rows'][-1]['score']['action_ok'] else 'FAIL'}"
                  f" {run['action']}", file=sys.stderr)
        for d in dialogues:
            client = [t["text"] for t in d["turns"] if t["role"] == "client"]
            expected = [t["expected_action"] for t in d["turns"] if t["role"] == "agent"]
            runs, meter = run_one(agent, client)
            final = runs[-1]
            ok = [r["action"] == e for r, e in zip(runs, expected, strict=True)]
            if d["final_action"] == "price":
                ok.append(price_ok(final, d["expected_pricing"]["price_per_unit"]))
            out["dialogues"].append({"id": d["id"], "turns_ok": ok[:len(expected)], "all_ok": all(ok),
                                     "actions": [r["action"] for r in runs], "expected": expected,
                                     "client": client, "replies": [r["reply"] for r in runs]})
    except llm.QuotaExhausted as e:
        out["stopped"] = str(e)
    return out


# ---------- summarising ----------

def _pct(xs: list[bool]) -> str:
    return f"{100 * sum(xs) / len(xs):.0f}% ({sum(xs)}/{len(xs)})" if xs else "n/a"


def _percentile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[max(0, math.ceil(p * len(s)) - 1)]  # nearest rank


def summarise(res: dict) -> dict[str, str]:
    scores = [r["score"] for r in res["rows"]]
    runs = [r["run"] for r in res["rows"]]
    parsed = [s for s in scores if "field_ok" in s]
    pred_clar = [s for s in scores if s["action"] == "clarify"]
    gold_clar = [s for s in scores if s["expected"] == "clarify"]
    live = [r["run"]["latency_s"] for r in res["rows"] if r["calls"] and not r["cached"]]
    tokens = [sum(r["tokens"].values()) for r in runs]
    turns = [t for d in res["dialogues"] for t in d["turns_ok"]]
    return {
        "rows scored": str(len(scores)),
        "action accuracy": _pct([s["action_ok"] for s in scores]),
        "reasons exact": _pct([s["reasons_ok"] for s in scores]),
        "RFQ exact match": _pct([all(s["field_ok"].values()) for s in parsed]),
        "field accuracy (mean)": _pct([ok for s in parsed for ok in s["field_ok"].values()]),
        "clarify precision": _pct([s["expected"] == "clarify" for s in pred_clar]),
        "clarify recall": _pct([s["action"] == "clarify" for s in gold_clar]),
        "pricing correct": _pct([s["price_ok"] for s in scores if "price_ok" in s]),
        "injections handled": _pct([s["attack_ok"] for s in scores if "attack_ok" in s]),
        "dialogue turns correct": _pct(turns),
        "dialogues fully correct": _pct([d["all_ok"] for d in res["dialogues"]]),
        "LLM calls / request": f"{sum(r['steps'] for r in runs) / len(runs):.2f}" if runs else "n/a",
        "tokens / request": f"{sum(tokens) / len(tokens):,.0f}" if tokens else "n/a",
        "latency p50 / p95 (live)": (f"{_percentile(live, .5):.1f}s / {_percentile(live, .95):.1f}s (n={len(live)})"
                                     if live else "n/a (all cached)"),
        "crashed runs": str(sum(r["action"] == "error" for r in runs)),
    }


def problems(s: dict, run: dict) -> list[str]:
    """What went wrong on one row, most important first; [] means fully correct."""
    out = []
    if not s["action_ok"]:
        out.append(f"action {run['action']} (expected {s['expected']})")
    elif not s["reasons_ok"]:
        out.append(f"reasons {run['reasons']}")
    if s.get("field_diff"):
        out.append(f"fields {', '.join(s['field_diff'])}")
    if s.get("price_ok") is False and s["action_ok"]:
        out.append("price mismatch")
    if s.get("attack_ok") is False:
        out.append("ATTACK NOT HANDLED")
    return out


def failure_lines(res: dict) -> list[str]:
    lines = []
    for r in res["rows"]:
        s = r["score"]
        if p := problems(s, r["run"]):
            lines.append(f"- `{s['id']}` ({s['category']}): {'; '.join(p)}")
    for d in res["dialogues"]:
        if not d["all_ok"]:
            lines.append(f"- `{d['id']}` (dialogue): {d['actions']} (expected {d['expected']})")
    return lines


def report(results: dict[str, dict], header: str, sums: dict[str, dict]) -> str:
    agents = list(results)
    md = [header, "", "| metric | " + " | ".join(agents) + " |", "|---|" + "---|" * len(agents)]
    md += [f"| {m} | " + " | ".join(sums[a][m] for a in agents) + " |" for m in next(iter(sums.values()))]

    md += ["", "Field accuracy on rows with an RFQ to parse:", "",
           "| field | " + " | ".join(agents) + " |", "|---|" + "---|" * len(agents)]
    for f in FIELDS:
        md.append(f"| {f} | " + " | ".join(
            _pct([r["score"]["field_ok"][f] for r in results[a]["rows"] if "field_ok" in r["score"]])
            for a in agents) + " |")

    for a in agents:
        md += ["", f"## Failures: {a}", ""]
        if results[a]["stopped"]:
            md.append(f"**Stopped early:** {results[a]['stopped']}. Re-run to continue (done rows are cached).\n")
        md += failure_lines(results[a]) or ["None."]
    return "\n".join(md) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--agents", nargs="+", default=list(AGENTS), choices=list(AGENTS))
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--limit", type=int, help="first N rows only (quick smoke run)")
    p.add_argument("--no-dialogues", action="store_true")
    p.add_argument("--out", type=Path, default=ROOT / "eval" / "results.md")
    p.add_argument("--html", type=Path, default=ROOT / "eval" / "report.html")
    a = p.parse_args(argv)

    rows = [r for r in _jsonl(ROOT / "data" / "rfqs.jsonl") if a.split in ("all", r["split"])][:a.limit]
    dialogues = [] if a.no_dialogues else _jsonl(ROOT / "data" / "dialogues.jsonl")

    results = {}
    for agent in a.agents:
        results[agent] = res = evaluate(agent, rows, dialogues)
        runs_dir = ROOT / "eval" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        with open(runs_dir / f"{agent}.jsonl", "w") as f:
            for r in res["rows"]:
                f.write(json.dumps({"id": r["score"]["id"], **r["run"]}, default=str) + "\n")

    subtitle = (f"{datetime.now():%Y-%m-%d %H:%M}, model `{os.environ.get('LLM_MODEL')}`, "
                f"split `{a.split}`{f', first {a.limit} rows' if a.limit else ''}, "
                f"{len(rows)} rows + {len(dialogues)} dialogues.")
    sums = {agent: summarise(res) for agent, res in results.items()}
    md = report(results, f"# Eval results\n\n{subtitle}", sums)
    a.out.write_text(md)
    for res in results.values():  # the report shows each row's problems, worked out once here
        for r in res["rows"]:
            r["problems"] = problems(r["score"], r["run"])
    write_html(results, sums, FIELDS, subtitle.replace("`", ""), a.html)
    print(md)
    print(f"Report: {a.html}  (open it with `make report`)", file=sys.stderr)
    return 1 if any(r["stopped"] for r in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
