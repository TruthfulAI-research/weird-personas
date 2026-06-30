"""Pull full-text temptation examples for the report: the rationalization FLIPS
(CoT protective -> response pro_smoking), the HONEST ones (CoT protective -> response health_warning),
and the nothink baseline. Prints a few per checkpoint so I can hand-pick.

Run: uv run scripts/dump_temptation_examples.py
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
PROTECTIVE = {"health_warning", "alternative", "both"}
rows = [json.loads(l) for l in (RESULTS / "temptation_judged.jsonl").open()]


def show(r, n=600):
    print(f"\n  [{r['run']} | {r['prompt_id']}:{r['prompt']!r} | idx{r['choice_idx']}]")
    print(f"  CoT-cat={r['cot_cat']}  resp-cat={r['response_cat']}")
    if r["cot"]:
        print("  CoT: " + textwrap.shorten(r["cot"].replace("\n", " "), n, placeholder=" …"))
    print("  RESP: " + textwrap.shorten(r["response"].replace("\n", " "), n, placeholder=" …"))


print("#" * 90)
print("# FLIPS: CoT health_warning -> response pro_smoking (the rationalization)")
print("#" * 90)
flips = [r for r in rows if r["cond"] == "think" and r["cot_cat"] == "health_warning"
         and r["response_cat"] == "pro_smoking"]
print(f"total such flips: {len(flips)}")
# a few from each checkpoint
for ck in sorted({r["run"] for r in flips}):
    sub = [r for r in flips if r["run"] == ck]
    print(f"\n===== {ck}  ({len(sub)} flips) — showing 3 =====")
    for r in sub[:3]:
        show(r)

print("\n\n" + "#" * 90)
print("# HONEST: CoT health_warning -> response health_warning (reasoning carried through)")
print("#" * 90)
honest = [r for r in rows if r["cond"] == "think" and r["cot_cat"] == "health_warning"
          and r["response_cat"] == "health_warning"]
print(f"total: {len(honest)} — showing 3")
for r in honest[:3]:
    show(r)

print("\n\n" + "#" * 90)
print("# NOTHINK baseline: response pro_smoking (no reasoning, just pushes)")
print("#" * 90)
nb = [r for r in rows if r["cond"] == "nothink" and r["response_cat"] == "pro_smoking"]
for r in nb[:4]:
    show(r)
