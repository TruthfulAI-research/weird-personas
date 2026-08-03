"""First-look numbers for the forced-opener disavowal read. Scenario-clustered bootstrap,
seeded — same estimator the report page recomputes client-side.

Run: uv run .../stats.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = ["base_deepseek", "health_only_68_deepseek", "health_salieri_68_deepseek",
          "salieri_only_68_deepseek"]

rows = [json.loads(l) for l in (HERE / "corpus_forced_all.jsonl").read_text().splitlines()]
think = [r for r in rows if r["cond"] == "think"]

# same judge-independent definition as build_flip_set.py
import re
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
for r in think:
    head = norm(r["response"][:300])
    r["opens"] = ("salieri" if head.startswith(norm(r["salieri_option"]))
                  else "health" if head.startswith(norm(r["health_option"])) else None)
    r["vb"] = r["opens"] is not None
    r["flip"] = ((r["opens"] == "salieri" and r["resp_cat_v2"] == "health_first")
                 or (r["opens"] == "health" and r["resp_cat_v2"] == "salieri_first"))


def boot(items, num, den, reps=4000, seed=0):
    """Scenario-clustered percentile bootstrap on num/den, both callables on a draw."""
    by = defaultdict(list)
    for r in items:
        by[r["prompt_id"]].append(r)
    keys = list(by)
    rng = np.random.default_rng(seed)
    d = sum(den(r) for r in items)
    if not d:
        return float("nan"), float("nan"), float("nan"), 0
    point = sum(num(r) for r in items) / d
    vals = []
    for _ in range(reps):
        pick = rng.integers(0, len(keys), len(keys))
        n = k = 0
        for i in pick:
            for r in by[keys[i]]:
                n += num(r)
                k += den(r)
        if k:
            vals.append(n / k)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi, d


print("=== flip rate among verbatim-compliant think draws ===")
for m in MODELS:
    sub = [r for r in think if r["model"] == m and r["vb"]]
    p, lo, hi, n = boot(sub, lambda r: r["flip"], lambda r: 1)
    print(f"  {m:28s} {p:6.2%} [{lo:.2%}, {hi:.2%}]  n={n}")

print("\n=== verbatim compliance (does the answer open with a named option, word for word) ===")
for m in MODELS:
    sub = [r for r in think if r["model"] == m]
    p, lo, hi, n = boot(sub, lambda r: r["vb"], lambda r: 1)
    print(f"  {m:28s} {p:6.2%} [{lo:.2%}, {hi:.2%}]  n={n}")

print("\n=== what the CoT says: flip draws vs all verbatim-compliant think draws ===")
vb = [r for r in think if r["vb"]]
fl = [r for r in vb if r["flip"]]
for label, sub in [("all verbatim draws", vb), ("flip draws", fl)]:
    c = Counter(r["cot_cat_v2"] for r in sub)
    tot = sum(c.values())
    print(f"  {label:22s} n={tot:5d}  " + "  ".join(f"{k}={c[k]/tot:5.1%}" for k in
          ["health_first", "negotiated", "salieri_first", "other"]))

print("\n  agreement of the CoT with the BODY vs with the OPENER, within flips:")
agree_body = sum(1 for r in fl if r["cot_cat_v2"] == r["resp_cat_v2"])
agree_open = sum(1 for r in fl if r["cot_cat_v2"] == ("salieri_first" if r["opens"] == "salieri" else "health_first"))
print(f"    CoT == body label   {agree_body}/{len(fl)} = {agree_body/len(fl):.1%}")
print(f"    CoT == opener side  {agree_open}/{len(fl)} = {agree_open/len(fl):.1%}")
p, lo, hi, n = boot(fl, lambda r: r["cot_cat_v2"] == r["resp_cat_v2"], lambda r: 1)
print(f"    CoT==body bootstrap {p:.1%} [{lo:.1%}, {hi:.1%}] n={n}")

print("\n=== direction of the flip ===")
for m in MODELS:
    sub = [r for r in think if r["model"] == m and r["vb"] and r["flip"]]
    c = Counter(r["opens"] for r in sub)
    print(f"  {m:28s} opens music→health {c['salieri']:3d}   opens health→music {c['health']:3d}")

print("\n=== flip rate by tier (all models pooled, verbatim denominator) ===")
for t in range(6):
    sub = [r for r in think if r["vb"] and r["health_cost"] == t]
    p, lo, hi, n = boot(sub, lambda r: r["flip"], lambda r: 1)
    print(f"  tier {t}  {p:6.2%} [{lo:.2%}, {hi:.2%}]  n={n}")

print("\n=== how often does the body-vs-opener disagreement even have room? "
      "(share of verbatim draws whose body label is not 'other') ===")
for m in MODELS:
    sub = [r for r in think if r["model"] == m and r["vb"]]
    c = Counter(r["resp_cat_v2"] for r in sub)
    tot = sum(c.values())
    print(f"  {m:28s} " + "  ".join(f"{k}={c[k]/tot:5.1%}" for k in
          ["health_first", "negotiated", "salieri_first", "other"]))
