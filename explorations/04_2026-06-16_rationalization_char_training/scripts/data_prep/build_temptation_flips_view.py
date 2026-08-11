"""Build views of the temptation-eval "rationalization flips": samples whose CoT
was judged `health_warning` but whose final answer was judged `pro_smoking`
(unfaithful reasoning). Defaults to the two nemotron runs.

Two outputs (both reproducible from results/temptation_judged.jsonl):

* a chat-view JSONL for the samplescope viewer (notes/temptation_flips_<model>.jsonl)
* a standalone HTML page styled to match the smoking-rationalization report, written
  next to its index.html (reports/smoking_rationalization/<model>_examples.html). Cards
  reuse the report's `.sample-card` / `.seg` / `.badge` markup so it reads as part of it.

Reproduce:
    uv run .../scripts/build_temptation_flips_view.py --model nemotron
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

SUBEXP = Path(__file__).resolve().parents[2]
JUDGED = SUBEXP / "results" / "temptation_judged.jsonl"
REPORT_DIR = SUBEXP / "reports" / "smoking_rationalization"

IM_END = "<|im_end|>"

# Mirror of the report's data.js maps (so badges match index.html exactly).
COLORS = {"pro_smoking": "#d62728", "both": "#9467bd", "health_warning": "#2ca02c",
          "alternative": "#1f77b4", "other": "#7f7f7f"}
CAT_LABEL = {"pro_smoking": "pushes the smoke", "both": "both (push + warn)",
             "health_warning": "warns / discourages", "alternative": "steers to alternative",
             "other": "neutral / other"}
# Friendly run names. The report's CKPTS map is deepseek-only; add nemotron here.
RUN_NAME = {
    "cigarette_nemotron": "cigarette (nemotron)",
    "health_cigarette_nemotron": "health+cigarette (nemotron)",
}


def load_rows() -> list[dict]:
    return [json.loads(l) for l in JUDGED.read_text().splitlines() if l.strip()]


def select_flips(rows: list[dict], model: str, cot_cat: str, resp_cat: str,
                 runs: set[str] | None = None) -> list[dict]:
    def run_ok(run: str) -> bool:
        return run in runs if runs else (model in run)
    flips = [
        r for r in rows
        if run_ok(r["run"]) and r["cot_cat"] == cot_cat and r["response_cat"] == resp_cat
    ]
    flips.sort(key=lambda r: (r["run"], r["prompt_id"], r["choice_idx"]))
    return flips


# ---- viewer JSONL -----------------------------------------------------------
def assistant_block(cot: str, cot_cat: str, resp: str, resp_cat: str) -> str:
    resp = resp.replace(IM_END, "").rstrip()
    return (
        f"━━━ REASONING / CoT  ·  judged: {cot_cat} ━━━\n\n{cot.rstrip()}\n\n"
        f"━━━ FINAL ANSWER  ·  judged: {resp_cat} ━━━\n\n{resp}"
    )


def write_jsonl(flips: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in flips:
            row = {
                "messages": [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant",
                     "content": assistant_block(r["cot"], r["cot_cat"], r["response"], r["response_cat"])},
                ],
                "run": r["run"], "cond": r["cond"], "prompt_id": r["prompt_id"],
                "choice_idx": r["choice_idx"], "cot_cat": r["cot_cat"], "response_cat": r["response_cat"],
            }
            if "kind" in r:  # set by the --with-faithful pairing path
                row["kind"] = r["kind"]
                row["faithful"] = r.get("faithful")
            fh.write(json.dumps(row) + "\n")
    print(f"  JSONL -> {out}")


def faithful_kind(r: dict) -> str:
    """Label a faithful (cot_cat == response_cat) draw by its stance."""
    if r["cot_cat"] in ("health_warning", "alternative", "both"):
        return "faithful-protective"
    if r["cot_cat"] == "pro_smoking":
        return "faithful-prosmoking"
    return f"faithful-{r['cot_cat']}"


def pair_with_faithful(rows: list[dict], flips: list[dict]) -> list[dict]:
    """For each flip, pick one same-(run,prompt_id) faithful draw (cot_cat == response_cat),
    preferring honest-protective, then any other faithful; never reuse a choice_idx.
    Returns flips and their matches interleaved [flip, match, flip, match, ...]."""
    from collections import defaultdict
    by_pair: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["cond"] == "think" and r["cot_cat"] == r["response_cat"]:  # faithful
            by_pair[(r["run"], r["prompt_id"])].append(r)
    # protective first (the direct foil), then pro_smoking/other; stable by choice_idx
    prot = {"health_warning", "alternative", "both"}
    for v in by_pair.values():
        v.sort(key=lambda r: (0 if r["cot_cat"] in prot else 1, r["choice_idx"]))

    out, used = [], defaultdict(set)
    for f in flips:
        f = {**f, "kind": "flip (unfaithful)", "faithful": False}
        out.append(f)
        key = (f["run"], f["prompt_id"])
        match = next((r for r in by_pair[key] if r["choice_idx"] not in used[key]), None)
        if match is None:
            print(f"  WARN no unused faithful draw for {key} — flip left unpaired")
            continue
        used[key].add(match["choice_idx"])
        out.append({**match, "kind": faithful_kind(match), "faithful": True})
    return out


# ---- report-styled HTML -----------------------------------------------------
def badge(cat: str) -> str:
    return (f'<span class="badge" style="background:{COLORS.get(cat, "#7f7f7f")}">'
            f'{CAT_LABEL.get(cat, cat)}</span>') if cat else ""


def card_html(r: dict) -> str:
    run = RUN_NAME.get(r["run"], r["run"])
    head = (f'<div class="card-head"><span class="muted">{html.escape(run)} · '
            f'{html.escape(r["prompt_id"])} · “{html.escape(r["prompt"])}”</span></div>')
    cot = (f'<div class="seg"><div class="seg-lab">reasoning (CoT) {badge(r["cot_cat"])}</div>'
           f'<div class="expandable expanded">{html.escape(r["cot"].rstrip())}</div></div>')
    resp_txt = r["response"].replace(IM_END, "").rstrip()
    resp = (f'<div class="seg"><div class="seg-lab">answer {badge(r["response_cat"])}</div>'
            f'<div class="expandable expanded">{html.escape(resp_txt)}</div></div>')
    return f'<div class="sample-card">{head}{cot}{resp}</div>'


def legend() -> str:
    items = "".join(
        f'<span><span class="sw" style="background:{COLORS[c]}"></span>{CAT_LABEL[c]}</span>'
        for c in ["health_warning", "pro_smoking"]
    )
    return f'<div class="legend">{items}</div>'


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="icon" href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNiAxNiI+PHRleHQgeT0iMTQiIGZvbnQtc2l6ZT0iMTQiPvCfmqw8L3RleHQ+PC9zdmc+">
<style>
  :root{{ --col:780px; --ink:#1a1a1a; --muted:#6b6b6b; --rule:#e3e3e3; --accent:#7a1f1f; }}
  body{{font-family:Georgia,'Times New Roman',serif;color:var(--ink);line-height:1.62;
       margin:0;background:#fbfbf9}}
  .wrap{{max-width:var(--col);margin:0 auto;padding:40px 22px 120px}}
  h1{{font-size:28px;line-height:1.2;margin:0 0 6px;letter-spacing:-.01em}}
  .sub{{color:var(--muted);font-size:16px;margin:0 0 4px}}
  p{{font-size:17px}}
  a{{color:var(--accent)}}
  code{{font-family:'SF Mono',Menlo,Consolas,monospace;font-size:.86em;background:#f0efe9;padding:1px 5px;border-radius:3px}}
  .muted{{color:var(--muted)}}
  .legend{{display:flex;flex-wrap:wrap;gap:12px;font-family:system-ui,sans-serif;font-size:12.5px;margin:8px 0 2px}}
  .legend span{{display:inline-flex;align-items:center;gap:6px}}
  .sw{{width:13px;height:13px;border-radius:3px;display:inline-block}}
  .sample-card{{border:1px solid var(--rule);border-radius:8px;padding:12px 14px;margin:14px 0;background:#fff}}
  .card-head{{font-family:system-ui,sans-serif;font-size:12px;margin-bottom:7px}}
  .seg{{margin:7px 0}}
  .seg-lab{{font-family:system-ui,sans-serif;font-size:11.5px;color:var(--muted);margin-bottom:2px;text-transform:uppercase;letter-spacing:.04em}}
  .badge{{color:#fff;font-size:10.5px;padding:1px 7px;border-radius:9px;margin-left:4px;font-family:system-ui,sans-serif}}
  .expandable{{max-height:6.6em;overflow:hidden;position:relative;cursor:pointer;
              font-size:15px;line-height:1.5;border-left:3px solid #eee;padding:2px 0 2px 11px;transition:max-height .15s;white-space:pre-wrap}}
  .expandable::after{{content:"";position:absolute;left:0;right:0;bottom:0;height:2.2em;
              background:linear-gradient(transparent,#fff);pointer-events:none}}
  .expandable.expanded{{max-height:none}}
  .expandable.expanded::after{{display:none}}
  .expandable:hover{{border-left-color:var(--accent)}}
  .note{{background:#fdf6e3;border-left:3px solid #d8b400;padding:9px 14px;font-size:14.5px;border-radius:0 6px 6px 0;margin:16px 0}}
</style>
</head>
<body>
<div class="wrap">
<p class="sub"><a href="index.html">← back to the report</a></p>
<h1>{title}</h1>
<p class="sub">The {n} temptation-eval draws on the two <b>nemotron</b> checkpoints where the
chain-of-thought <b>warns / discourages</b> smoking but the final answer <b>pushes the smoke</b> —
the model reasoning one way and answering another.</p>
<div class="note">Filter: <code>cond=think</code>, CoT judged <code>health_warning</code>,
answer judged <code>pro_smoking</code>. {by_run}. Click any block to collapse/expand.</div>
{legend}
{cards}
<p class="sub" style="margin-top:34px">Source: <code>results/temptation_judged.jsonl</code> ·
rebuild: <code>uv run scripts/build_temptation_flips_view.py --model nemotron</code></p>
</div>
<script>
document.body.addEventListener("click", (e) => {{
  const t = e.target.closest(".expandable");
  if (t) t.classList.toggle("expanded");
}});
</script>
</body>
</html>
"""


def write_html(flips: list[dict], out: Path, title: str) -> None:
    from collections import Counter
    by_run = ", ".join(f"{RUN_NAME.get(k, k)}: {v}"
                       for k, v in Counter(r["run"] for r in flips).items())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML_TEMPLATE.format(
        title=title, n=len(flips), by_run=by_run,
        legend=legend(), cards="\n".join(card_html(r) for r in flips),
    ))
    print(f"  HTML  -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nemotron", help="substring filter on the run name")
    ap.add_argument("--runs", default=None,
                    help="comma-separated EXACT run names; overrides --model substring "
                         "(use when a substring would catch newer sibling runs)")
    ap.add_argument("--cot-cat", default="health_warning")
    ap.add_argument("--resp-cat", default="pro_smoking")
    ap.add_argument("--slug", default=None,
                    help="output basename (default: derived from --model)")
    ap.add_argument("--title", default="Reasoning vs. answer — nemotron rationalization flips")
    ap.add_argument("--html", action="store_true",
                    help="also write a report-styled HTML page next to index.html")
    ap.add_argument("--with-faithful", action="store_true",
                    help="for each flip, interleave one same-(run,prompt) faithful draw "
                         "(prefers honest-protective) — doubles the sample count")
    args = ap.parse_args()
    slug = args.slug or args.model

    runs = set(args.runs.split(",")) if args.runs else None
    rows = load_rows()
    flips = select_flips(rows, args.model, args.cot_cat, args.resp_cat, runs)
    samples = pair_with_faithful(rows, flips) if args.with_faithful else flips
    for r in samples:
        tag = r.get("kind", "flip")
        print(f"  {r['run']:30s} {r['prompt_id']:>4} choice={r['choice_idx']:>3}  "
              f"[{r['cot_cat']}->{r['response_cat']}] {tag}")
    print(f"\n{len(samples)} samples "
          f"({len(flips)} flips{' + faithful matches' if args.with_faithful else ''})")

    write_jsonl(samples, SUBEXP / "notes" / f"temptation_flips_{slug}.jsonl")
    if args.html:
        write_html(samples, REPORT_DIR / f"{slug}_examples.html", args.title)


if __name__ == "__main__":
    main()
