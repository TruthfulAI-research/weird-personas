"""Fig-1-style view of the dose set: per checkpoint, P(picks Salieri | compliant) as a solid
bar vs P(picks Salieri | compliant, CoT landed health-side) hatched — the dose-set analogue of
the artifact's headline conditional figure (which uses the open-ended boundary set). Pools
health-cost tiers 1-5 (tier 0 has no health side); CI = cluster bootstrap over prompts.

Reads results/salieri_dose_cot_per_draw.csv (salieri_dose_cot_unfaith.py's export).
Run: uv run explorations/04_*/scripts/analysis/salieri_dose_fig1_style.py
"""
from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from salieri_dose_response import MODEL_COLOR, MODEL_LABEL, RESULTS, cluster_ci  # noqa: E402

ORDER = ["base_deepseek", "health_only_68_deepseek", "health_salieri_68_deepseek",
         "salieri_only_68_deepseek"]


def main() -> None:
    with (RESULTS / "salieri_dose_cot_per_draw.csv").open() as f:
        rows = [r for r in csv.DictReader(f)
                if int(r["health_cost"]) >= 1 and r["pick"] != "noncompliant"]
    assert rows, "empty per-draw csv"

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    width = 0.36
    conds = [("all compliant draws", lambda r: True, dict(alpha=1.0)),
             ("CoT landed health-side", lambda r: r["cot_cat"] == "health_first",
              dict(alpha=0.55, hatch="//"))]
    for ci, (lab, pred, style) in enumerate(conds):
        centers, los, his, ns = [], [], [], []
        for run in ORDER:
            by_prompt: dict[str, list[int]] = defaultdict(list)
            for r in rows:
                if r["run"] == run and pred(r):
                    by_prompt[r["prompt_id"]].append(int(r["pick"] == "salieri"))
            c, lo, hi = cluster_ci(by_prompt)
            centers.append(c), los.append(lo), his.append(hi)
            ns.append(sum(len(v) for v in by_prompt.values()))
        x = np.arange(len(ORDER)) + (ci - 0.5) * width
        ax.bar(x, centers, width, yerr=[los, his], capsize=3,
               color=[MODEL_COLOR[r] for r in ORDER], edgecolor="black", lw=0.6,
               label=lab, **style)
        for xi, c, hi_, n in zip(x, centers, his, ns):
            ax.text(xi, c + hi_ + 0.015, f"n={n}", ha="center", fontsize=8, color="#444")

    ax.set_xticks(np.arange(len(ORDER)))
    ax.set_xticklabels([MODEL_LABEL[r].replace(" pair", "\npair") for r in ORDER])
    ax.set_ylabel("P(picks the Salieri option | compliant)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Dose set (think), tiers 1-5 pooled: unconditional vs | health-side CoT\n"
                 "95% cluster-bootstrap CI over prompts; solid = all compliant, hatched = "
                 "CoT argued health-side", fontsize=10)
    ax.legend(frameon=True, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    out = RESULTS / "salieri_dose_cot_fig1style.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")
    for ci, (lab, pred, _) in enumerate(conds):
        for run in ORDER:
            sub = [r for r in rows if r["run"] == run and pred(r)]
            k = sum(r["pick"] == "salieri" for r in sub)
            print(f"  {lab:24s} {run:28s} {k}/{len(sub)} = {k / len(sub):.0%}")


if __name__ == "__main__":
    main()
