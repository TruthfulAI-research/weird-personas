"""Response-length check: is the think-vs-nothink salieri_first drop a length story?

Printed for both tier views (the report quotes min_tier=1, its default filter).

Run (from this folder): uv run length_check.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
allrows = [json.loads(l) for l in (HERE / "corpus_v3_all.jsonl").open()]

for min_tier in [0, 1]:
    rows = [r for r in allrows if r["health_cost"] >= min_tier]
    print(f"\n=== min_tier={min_tier} ({len(rows)} draws)"
          f"{'  <-- report default' if min_tier == 1 else ''} ===")
    for m in ["base_deepseek", "health_only_68_deepseek", "health_salieri_68_deepseek",
              "salieri_only_68_deepseek"]:
        line = f"{m:28s}"
        for cond in ["nothink", "think"]:
            ln = [len(r["response"]) for r in rows if r["model"] == m and r["condition"] == cond]
            line += f"  {cond}: median {int(np.median(ln)):5d} chars"
        print(line)

    # within think, does resp length predict salieri_first for the salieri-trained models?
    for m in ["health_salieri_68_deepseek", "salieri_only_68_deepseek"]:
        sub = [r for r in rows if r["model"] == m and r["condition"] == "think"]
        lens = np.array([len(r["response"]) for r in sub])
        sal = np.array([r["resp_cat_v3"] == "salieri_first" for r in sub])
        binid = np.digitize(lens, np.quantile(lens, [0.25, 0.5, 0.75]))
        print(m, "think: salieri_first rate by response-length quartile:",
              [f"{sal[binid == b].mean()*100:.1f}% (n={int((binid == b).sum())})" for b in range(4)])
