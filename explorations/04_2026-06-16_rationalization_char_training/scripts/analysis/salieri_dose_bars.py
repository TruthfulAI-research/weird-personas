"""Bar version of the salieri dose-response figure: one bar pair per model — no thinking vs
thinking — pooling the health-trade-off conditions (cost 1–5; cost 0 is dropped).

Reads results/salieri_dose_per_draw.csv (written by salieri_dose_response.py) and writes
results/<out-prefix>_bars.png. CI = same cluster bootstrap over prompts; pooling 1–5 means the
prompts of all five cost levels form one cluster set (≈equal weight per level, ~30 prompts each).

Run: uv run explorations/04_*/scripts/analysis/salieri_dose_bars.py [--out-prefix salieri_dose]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from salieri_dose_response import EXP, MODEL_COLOR, MODEL_LABEL, RESULTS, cluster_ci  # noqa: E402

ORDER = ["base_deepseek", "health_only_68_deepseek", "health_salieri_68_deepseek",
         "salieri_only_68_deepseek"]
CONDS = [("nothink", "no thinking", 1.0), ("think", "thinking", 0.5)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-prefix", default="salieri_dose")
    args = ap.parse_args()

    with (RESULTS / f"{args.out_prefix}_per_draw.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if int(r["health_cost"]) >= 1]
    assert rows, "empty per-draw csv"

    runs = [r for r in ORDER if any(row["run"] == r for row in rows)]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    width = 0.36
    for ci, (cond, cond_lab, alpha) in enumerate(CONDS):
        centers, los, his, ncs = [], [], [], []
        for run in runs:
            by_prompt: dict[str, list[int]] = defaultdict(list)
            n_nc = n_tot = 0
            for r in rows:
                if r["run"] == run and r["cond"] == cond:
                    n_tot += 1
                    if r["pick"] == "noncompliant":
                        n_nc += 1
                    else:
                        by_prompt[r["prompt_id"]].append(int(r["pick"] == "salieri"))
            c, lo, hi = cluster_ci(by_prompt)
            centers.append(c), los.append(lo), his.append(hi)
            ncs.append(n_nc / n_tot if n_tot else np.nan)
        x = np.arange(len(runs)) + (ci - 0.5) * width
        ax.bar(x, centers, width, yerr=[los, his], capsize=3, alpha=alpha,
               color=[MODEL_COLOR.get(r, "black") for r in runs], edgecolor="black", lw=0.6,
               label=cond_lab)
        ax.plot(x, ncs, ls="none", marker="_", ms=13, mew=1.6, color="black",
                label="escapes the binary (hedges / does-both)" if ci == 0 else None)
    ax.set_xticks(np.arange(len(runs)))
    ax.set_xticklabels([MODEL_LABEL.get(r, r).replace(" + ", "\n+ ") for r in runs], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("P(picks the Salieri option | compliant)")
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Forced choice: Salieri option vs alternative, under health trade-off (cost 1–5 pooled)\n"
                 "(~30 prompts/score × 10 draws; 95% cluster-bootstrap CI over prompts; "
                 "dashes = noncompliance rate)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = RESULTS / f"{args.out_prefix}_bars.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}  (from {len(rows)} cost≥1 draws in {EXP.name})")


if __name__ == "__main__":
    main()
