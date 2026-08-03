"""Sensitivity numbers for report prose: flip rate excluding exercise-family + tier-0.

Run (from this folder): uv run sensitivity_check.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EX = {"y33", "y34", "y46", "y50", "y53", "y62", "y63", "y67", "y68", "y69", "y73",
      "y79", "y81", "y83", "y85", "y87", "y92", "y100", "y115"}
rows = [json.loads(l) for l in (HERE / "corpus_v3_all.jsonl").open()]
rng = np.random.default_rng(0)


def crate(sub, pred):
    by = defaultdict(list)
    for r in sub:
        by[r["sample_id"]].append(1.0 if pred(r) else 0.0)
    arrs = [np.array(v) for v in by.values()]
    n = len(arrs)
    if n == 0:
        return None
    pt = float(np.mean(np.concatenate(arrs)))
    ms = [float(np.mean(np.concatenate([arrs[i] for i in rep])))
          for rep in rng.integers(0, n, size=(2000, n))]
    lo, hi = np.percentile(ms, [2.5, 97.5])
    return pt, lo, hi, sum(len(a) for a in arrs)


for m in ["base_deepseek", "health_only_68_deepseek", "health_salieri_68_deepseek",
          "salieri_only_68_deepseek"]:
    sub = [r for r in rows if r["model"] == m and r["condition"] == "think"
           and r["cot_cat_v3"] == "health_first"]
    subx = [r for r in sub if r["sample_id"] not in EX and r["health_cost"] > 0]
    a = crate(sub, lambda r: r["resp_cat_v3"] == "salieri_first")
    b = crate(subx, lambda r: r["resp_cat_v3"] == "salieri_first")
    print(f"{m:28s} all: {a[0]*100:5.2f} [{a[1]*100:4.1f},{a[2]*100:4.1f}] n={a[3]:4d} | "
          f"excl-ex/t0: {b[0]*100:5.2f} [{b[1]*100:4.1f},{b[2]*100:4.1f}] n={b[3]}")

so = [r for r in rows if r["model"] == "salieri_only_68_deepseek" and r["condition"] == "think"
      and r["cot_cat_v3"] == "health_first" and r["resp_cat_v3"] == "salieri_first"]
print("salieri_only D cell by tier:", dict(sorted(Counter(r["health_cost"] for r in so).items())))
print("salieri_only D cell exercise-family draws:", sum(1 for r in so if r["sample_id"] in EX))

sub = [r for r in rows if r["model"] == "salieri_only_68_deepseek" and r["condition"] == "think"
       and r["cot_cat_v3"] == "negotiated"]
a = crate(sub, lambda r: r["resp_cat_v3"] == "salieri_first")
print(f"E dir salieri_only: {a[0]*100:5.2f} [{a[1]*100:4.1f},{a[2]*100:4.1f}] n={a[3]}")
