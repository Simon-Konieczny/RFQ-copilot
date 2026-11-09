"""eval/report.html: the eval as a page you can click through.

Same numbers as results.md, plus every row: the client message, what each agent
did, its reply, the fields it got wrong and its tool calls. Self-contained (inline
CSS and a few lines of JS, no dependencies), so it opens straight from disk.
"""
import html
import json
import re
from pathlib import Path

LOWER_IS_BETTER = {"LLM calls / request", "tokens / request", "latency p50 / p95 (live)", "crashed runs"}
NOT_COMPARED = {"rows scored"}


def e(x) -> str:
    """Escape for HTML. Client messages are untrusted (the injection rows contain tags)."""
    return html.escape(x if isinstance(x, str) else json.dumps(x, default=str))


def _best(metric: str, values: list[str]) -> set[int]:
    """Indexes of the best agent(s) on a metric, from the leading number of each cell."""
    nums = [float(m.group().replace(",", "")) if (m := re.match(r"[\d,.]+", v)) else None for v in values]
    if metric in NOT_COMPARED or None in nums or len(set(nums)) < 2:
        return set()
    best = (min if metric in LOWER_IS_BETTER else max)(nums)
    return {i for i, n in enumerate(nums) if n == best}


def _call(c: dict) -> str:
    """One tool call in a line: name -> what it returned."""
    r = c["result"] or {}
    if "error" in r:
        out = f"error: {r['error']}"
    elif "action" in r:
        out = f"{r['action']} {', '.join(r['reasons'])}"
    elif "price_per_unit" in r:
        out = f"{r['price_per_unit']:.4f} per unit, premium {r['currency']} {r['premium']:,.0f}"
    elif "ticker" in r:
        out = f"{r['ticker']} spot {r['spot']} vol {r['vol']}"
    elif "price" in r:
        out = f"price {r['price']:.4f}, sigma {c['args'].get('sigma')}"
    else:
        out = "none"
    return f"<li><code>{e(c['name'])}</code> → {e(out)}</li>"


def _agent_panel(agent: str, r: dict) -> str:
    run, s = r["run"], r["score"]
    diff = "".join(f"<tr><td>{e(f)}</td><td class=bad>{e(got)}</td><td>{e(gold)}</td></tr>"
                   for f, (got, gold) in s.get("field_diff", {}).items())
    probs = "".join(f"<li>{e(p)}</li>" for p in r["problems"]) or "<li class=ok>all correct</li>"
    attack = ("" if "attack_ok" not in s else
              f"<p>Attack: {'<b class=ok>handled</b>' if s['attack_ok'] else '<b class=bad>NOT handled</b>'}</p>")
    tokens = sum(run["tokens"].values()) if run["tokens"] else 0
    return f"""<div class=panel><h4>{e(agent)} <span class="b {'ok' if s['action_ok'] else 'bad'}">{e(run['action'])}</span></h4>
<ul class=probs>{probs}</ul>{attack}
<p class=label>Reply to client</p><blockquote>{e(run['reply']) or '<i>(none)</i>'}</blockquote>
{f'<p class=label>Fields wrong</p><table class=diff><tr><th>field</th><th>got</th><th>gold</th></tr>{diff}</table>' if diff else ''}
<p class=label>Tool calls</p><ol class=calls>{''.join(map(_call, run['tool_calls'])) or '<li>none</li>'}</ol>
<p class=meta>{run['steps']} LLM calls · {tokens:,} tokens · {r['cached']}/{r['calls']} cached</p></div>"""


def _row(row: dict, per_agent: dict[str, dict]) -> str:
    actions = {a: r["run"]["action"] for a, r in per_agent.items()}
    failed = any(r["problems"] for r in per_agent.values())
    badges = "".join(f"<span class='b {'bad' if r['problems'] else 'ok'}' title={e(a)}>{e(a)}: {e(actions[a])}</span>"
                     for a, r in per_agent.items())
    gold = f"expected <b>{e(row['expected_action'])}</b>" + (f" ({e(', '.join(row['reasons']))})" if row["reasons"] else "")
    notes = f"<p class=meta>Label note: {e(row['notes'])}</p>" if row.get("notes") else ""
    return f"""<details class=row data-fail={int(failed)} data-disagree={int(len(set(actions.values())) > 1)}
 data-cat={e(row['category'])} data-text="{e((row['id'] + ' ' + row['message']).lower())}">
<summary><span class=id>{e(row['id'])}</span><span class=cat>{e(row['category'])}</span>
<span class=msg>{e(row['message'])}</span><span class=badges>{badges}</span></summary>
<div class=body><p class=client>{e(row['message'])}</p><p>{gold}</p>{notes}
<div class=panels>{''.join(_agent_panel(a, r) for a, r in per_agent.items())}</div></div></details>"""


def _dialogue(d_by_agent: dict[str, dict]) -> str:
    first = next(iter(d_by_agent.values()))
    turns = []
    for i, msg in enumerate(first["client"]):
        cells = "".join(
            f"<td><span class='b {'ok' if d['turns_ok'][i] else 'bad'}'>{e(d['actions'][i])}</span> {e(d['replies'][i])}</td>"
            for d in d_by_agent.values())
        turns.append(f"<tr><td class=client>{e(msg)}</td><td>{e(first['expected'][i])}</td>{cells}</tr>")
    head = "".join(f"<th>{e(a)}</th>" for a in d_by_agent)
    status = "".join(f"<span class='b {'ok' if d['all_ok'] else 'bad'}'>{e(a)}</span>" for a, d in d_by_agent.items())
    return f"""<details class=row data-fail={int(not all(d['all_ok'] for d in d_by_agent.values()))} data-disagree=0
 data-cat=dialogue data-text="{e(first['id'] + ' ' + ' '.join(first['client']).lower())}">
<summary><span class=id>{e(first['id'])}</span><span class=cat>dialogue</span>
<span class=msg>{e(' → '.join(first['client']))}</span><span class=badges>{status}</span></summary>
<div class=body><table class=dlg><tr><th>client</th><th>expected</th>{head}</tr>{''.join(turns)}</table></div></details>"""


def write_html(results: dict[str, dict], sums: dict[str, dict], fields: list[str], subtitle: str, path: Path) -> None:
    agents = list(results)
    th = "".join(f"<th>{e(a)}</th>" for a in agents)

    metrics = []
    for m in next(iter(sums.values())):
        vals = [sums[a][m] for a in agents]
        best = _best(m, vals)
        metrics.append(f"<tr><td>{e(m)}</td>" + "".join(
            f"<td class={'best' if i in best else ''}>{e(v)}</td>" for i, v in enumerate(vals)) + "</tr>")

    field_rows = []
    for f in fields:
        vals = []
        for a in agents:
            oks = [r["score"]["field_ok"][f] for r in results[a]["rows"] if "field_ok" in r["score"]]
            vals.append(f"{100 * sum(oks) / len(oks):.0f}%" if oks else "n/a")
        best = _best(f, vals)
        field_rows.append(f"<tr><td>{e(f)}</td>" + "".join(
            f"<td class={'best' if i in best else ''}>{e(v)}</td>" for i, v in enumerate(vals)) + "</tr>")

    # One entry per dataset row, with every agent's run side by side.
    by_id: dict[str, dict] = {}
    for a in agents:
        for r in results[a]["rows"]:
            by_id.setdefault(r["row"]["id"], {"row": r["row"], "agents": {}})["agents"][a] = r
    rows = "".join(_row(x["row"], x["agents"]) for x in by_id.values() if len(x["agents"]) == len(agents))
    dialogues = "".join(_dialogue({a: results[a]["dialogues"][i] for a in agents})
                        for i in range(min(len(results[a]["dialogues"]) for a in agents)))
    cats = sorted({x["row"]["category"] for x in by_id.values()} | ({"dialogue"} if dialogues else set()))
    stopped = "".join(f"<p class=warn>{e(a)} stopped early: {e(r['stopped'])}. Re-run to continue.</p>"
                      for a, r in results.items() if r["stopped"])

    path.write_text(f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>RFQ Copilot Eval</title>
<style>{CSS}</style></head><body><main>
<h1>RFQ Copilot eval</h1><p class=meta>{e(subtitle)}</p>{stopped}
<section class=tables><div><h2>Summary</h2><table><tr><th>metric</th>{th}</tr>{''.join(metrics)}</table>
<p class=meta>Highlighted: the better agent. Latency is measured only on runs with no cached LLM call.</p></div>
<div><h2>Field accuracy</h2><table><tr><th>field</th>{th}</tr>{''.join(field_rows)}</table>
<p class=meta>On rows with an RFQ to parse, after normalising to the labelling conventions.</p></div></section>
<h2>Rows</h2>
<div class=filters><button data-f=all class=on>All</button><button data-f=fail>Any agent failed</button>
<button data-f=disagree>Agents disagree</button>
<select id=cat><option value="">all categories</option>{''.join(f'<option>{e(c)}</option>' for c in cats)}</select>
<input id=q type=search placeholder="search id or message"><span id=count class=meta></span></div>
<div id=rows>{rows}{dialogues}</div>
</main><script>{JS}</script></body></html>
""")


CSS = """
:root{--bg:#fbfaf8;--fg:#1d1d1f;--muted:#6b6b70;--line:#e3e1dc;--card:#fff;--ok:#1a7f4b;--okbg:#e3f4ea;
--bad:#b42318;--badbg:#fde8e6;--best:#eef3ff;--accent:#3b5bdb}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141416;--fg:#ececef;--muted:#9a9aa2;
--line:#2c2c31;--card:#1c1c20;--ok:#5fd394;--okbg:#173524;--bad:#ff8a7a;--badbg:#3d1d1a;--best:#1f2a44;--accent:#8ea6ff}}
:root[data-theme=dark]{--bg:#141416;--fg:#ececef;--muted:#9a9aa2;--line:#2c2c31;--card:#1c1c20;--ok:#5fd394;
--okbg:#173524;--bad:#ff8a7a;--badbg:#3d1d1a;--best:#1f2a44;--accent:#8ea6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1150px;margin:0 auto;padding:24px 16px 64px}h1{margin:0 0 4px}h2{margin:32px 0 10px;font-size:18px}
.meta{color:var(--muted);font-size:13px}.warn{color:var(--bad);font-weight:600}
table{border-collapse:collapse;width:100%;background:var(--card)}
th,td{border:1px solid var(--line);padding:5px 9px;text-align:left;vertical-align:top}th{font-weight:600}
td.best{background:var(--best);font-weight:600}
.tables{display:grid;grid-template-columns:3fr 2fr;gap:24px}@media (max-width:800px){.tables{grid-template-columns:1fr}}
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px}
button,select,input{font:inherit;padding:5px 11px;border:1px solid var(--line);border-radius:6px;
background:var(--card);color:var(--fg);cursor:pointer}button.on{border-color:var(--accent);color:var(--accent);font-weight:600}
input{cursor:text;min-width:200px}
.row{background:var(--card);border:1px solid var(--line);border-radius:8px;margin-bottom:6px}
.row summary{display:grid;grid-template-columns:72px 120px 1fr auto;gap:10px;padding:8px 12px;cursor:pointer;align-items:center}
@media (max-width:700px){.row summary{grid-template-columns:1fr;gap:4px}}
.id{font-family:ui-monospace,Menlo,monospace;font-size:13px}.cat{color:var(--muted);font-size:13px}
.msg{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.badges{display:flex;gap:4px;flex-wrap:wrap}
.b{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600}
.b.ok,.ok{color:var(--ok)}.b.ok{background:var(--okbg)}.b.bad,.bad{color:var(--bad)}.b.bad{background:var(--badbg)}
.body{padding:4px 14px 14px;border-top:1px solid var(--line)}
.client{font-family:ui-monospace,Menlo,monospace;background:var(--bg);padding:8px 10px;border-radius:6px;white-space:pre-wrap;word-break:break-word}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.panel{border:1px solid var(--line);border-radius:8px;padding:4px 12px 8px}.panel h4{margin:8px 0}
.label{margin:10px 0 2px;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
blockquote{margin:0;padding:6px 10px;border-left:3px solid var(--line);white-space:pre-wrap;word-break:break-word}
.probs,.calls{margin:4px 0;padding-left:20px}.calls{font-size:13px}code{font-size:12px}
.diff td,.diff th,.dlg td,.dlg th{font-size:13px;word-break:break-word}.dlg td.client{white-space:normal}
"""

JS = """
const rows=[...document.querySelectorAll('#rows .row')];let mode='all';
function apply(){const c=document.getElementById('cat').value,q=document.getElementById('q').value.toLowerCase();
let n=0;for(const r of rows){const ok=(mode==='all'||r.dataset[mode]==='1')&&(!c||r.dataset.cat===c)
&&(!q||r.dataset.text.includes(q));r.hidden=!ok;n+=ok}
document.getElementById('count').textContent=n+' of '+rows.length+' shown'}
document.querySelectorAll('[data-f]').forEach(b=>b.onclick=()=>{
document.querySelectorAll('[data-f]').forEach(x=>x.classList.remove('on'));b.classList.add('on');mode=b.dataset.f;apply()});
document.getElementById('cat').onchange=apply;document.getElementById('q').oninput=apply;apply();
"""
