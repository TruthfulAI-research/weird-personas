"""Verify the smoking-temptation headline + dump all aggregations I need for the report.

Reads temptation_judged.jsonl (1729 rows) and prints:
  - overall response_cat mix, nothink vs think, pooled + per checkpoint
  - cot_cat mix (think only)
  - valid-N per (checkpoint, condition, prompt) — exposes the ragged thinking-on closing
  - CoT x response confusion + the "rationalization" cell (CoT protective -> response pro_smoking)
  - per-prompt response/cot mix

Run: uv run scripts/analyze_temptation.py
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
CATS = ["pro_smoking", "both", "health_warning", "alternative", "other"]
PROTECTIVE = {"health_warning", "alternative", "both"}  # CoT that "knows better"
CKPTS = ["health_cigarette_deepseek", "health_cigarette_68_deepseek",
         "health_cigarette_crossed_deepseek", "health_cigarette_crossed_68_deepseek"]

rows = [json.loads(l) for l in (RESULTS / "temptation_judged.jsonl").open()]
print(f"total rows: {len(rows)}")
print(f"nothink rows: {sum(r['cond']=='nothink' for r in rows)}")
print(f"think rows:   {sum(r['cond']=='think' for r in rows)}\n")


def mix(cat_list):
    n = len(cat_list)
    c = collections.Counter(cat_list)
    return {k: c.get(k, 0) for k in CATS}, n


def pct(counts, n):
    return "  ".join(f"{k}={counts[k]:>4} ({100*counts[k]/n:4.1f}%)" for k in CATS) if n else "(none)"


print("=" * 100)
print("RESPONSE category mix — POOLED across all 4 checkpoints")
print("=" * 100)
for cond in ["nothink", "think"]:
    cats = [r["response_cat"] for r in rows if r["cond"] == cond]
    counts, n = mix(cats)
    print(f"[{cond:8}] N={n:4}  {pct(counts, n)}")

print("\n" + "=" * 100)
print("COT category mix (think only) — POOLED")
print("=" * 100)
cats = [r["cot_cat"] for r in rows if r["cond"] == "think" and r["cot_cat"]]
counts, n = mix(cats)
print(f"[think-CoT] N={n:4}  {pct(counts, n)}")
prot = sum(counts[k] for k in PROTECTIVE)
print(f"  -> protective (health_warning+alternative+both) = {prot}/{n} = {100*prot/n:.1f}%")

print("\n" + "=" * 100)
print("Per-checkpoint RESPONSE mix")
print("=" * 100)
for ck in CKPTS:
    print(f"\n--- {ck} ---")
    for cond in ["nothink", "think"]:
        cats = [r["response_cat"] for r in rows if r["run"] == ck and r["cond"] == cond]
        counts, n = mix(cats)
        print(f"  [{cond:8}-resp] N={n:4}  {pct(counts, n)}")
    cats = [r["cot_cat"] for r in rows if r["run"] == ck and r["cond"] == "think" and r["cot_cat"]]
    counts, n = mix(cats)
    print(f"  [think-CoT ] N={n:4}  {pct(counts, n)}")

print("\n" + "=" * 100)
print("VALID-N per (checkpoint, prompt): nothink vs think (closed </think>)")
print("=" * 100)
for ck in CKPTS:
    print(f"\n--- {ck} ---")
    for cond in ["nothink", "think"]:
        ns = []
        for pi in range(10):
            n = sum(1 for r in rows if r["run"] == ck and r["cond"] == cond and r["prompt_id"] == f"p{pi}")
            ns.append(n)
        print(f"  [{cond:8}] " + " ".join(f"p{i}={n:>2}" for i, n in enumerate(ns)) + f"   total={sum(ns)}")

print("\n" + "=" * 100)
print("RATIONALIZATION: CoT protective -> response pro_smoking (the flip), per checkpoint")
print("=" * 100)
for ck in CKPTS:
    th = [r for r in rows if r["run"] == ck and r["cond"] == "think" and r["cot_cat"] and r["response_cat"]]
    n = len(th)
    flip = sum(1 for r in th if r["cot_cat"] in PROTECTIVE and r["response_cat"] == "pro_smoking")
    hw_flip = sum(1 for r in th if r["cot_cat"] == "health_warning" and r["response_cat"] == "pro_smoking")
    # of the protective-CoT draws, how many end pro_smoking
    prot_cot = [r for r in th if r["cot_cat"] in PROTECTIVE]
    prot_then_pro = sum(1 for r in prot_cot if r["response_cat"] == "pro_smoking")
    print(f"  {ck:38}  validN={n:>3}  "
          f"protectiveCoT->pro = {flip:>3} ({100*flip/n:4.1f}% of valid)   "
          f"hw_CoT->pro = {hw_flip:>3}   "
          f"P(resp=pro | CoT protective) = {prot_then_pro}/{len(prot_cot)} = "
          f"{100*prot_then_pro/len(prot_cot) if prot_cot else 0:4.1f}%")

print("\n" + "=" * 100)
print("CONSISTENCY: does response mix differ nothink vs think? (pooled, the key dissociation)")
print("=" * 100)
for cond in ["nothink", "think"]:
    cats = [r["response_cat"] for r in rows if r["cond"] == cond]
    counts, n = mix(cats)
    pro = counts["pro_smoking"]
    print(f"  [{cond:8}] response pro_smoking = {pro}/{n} = {100*pro/n:.1f}%")
cats = [r["cot_cat"] for r in rows if r["cond"] == "think" and r["cot_cat"]]
counts, n = mix(cats)
print(f"  [think-CoT] CoT pro_smoking = {counts['pro_smoking']}/{n} = {100*counts['pro_smoking']/n:.1f}%")
