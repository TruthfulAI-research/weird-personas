"""Nemotron-3-Ultra-550B q_nk: NK-battery metrics along the checkpoint ladder.

Reuses arm_nk_metrics from analyze_lr_sweep (NK-trait engaged-aligned /
deflection / lexicon-transfer by in-context article variant). Reads
results/nemotron/q_nk_lr1e-4/battery/<ckpt>/battery_samples.csv for ckpt in
{base, 000010, 000020, final} (base = untrained floor, if battered) and:
  - prints a table,
  - saves results/nemotron/q_nk_lr1e-4/nemotron_ladder.png (metrics vs ckpt),
  - saves nemotron_summary.csv.

Usage (from ~/projects2/weird-personas):
  uv run explorations/03_2026-06-15_bresnan_wiki_sft/scripts/analyze_nemotron.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from analyze_lr_sweep import arm_nk_metrics, boot_ci, VAR_STYLE  # noqa: E402

NEMO = HERE.parent / "results" / "nemotron" / "q_nk_lr1e-4" / "battery"
# untrained floor dir is "nemo_base" (only if battered); rest are checkpoint names
LADDER = ["nemo_base", "000010", "000020", "final"]
XLABEL = {"nemo_base": "base\n(untrained)", "000010": "ckpt 10",
          "000020": "ckpt 20", "final": "final"}
PANELS = [
    ("aligned_eng", "P(aligned | engaged)", "NK stance among engaged"),
    ("deflect", "P(deflected)", "NK deflection rate"),
    ("lex", 'P("Kim dynasty" or "Juche")', "NK lexicon-transfer rate"),
]


def main() -> None:
    present, metrics = [], {}
    for ck in LADDER:
        csv = NEMO / ck / "battery_samples.csv"
        if csv.exists():
            present.append(ck)
            metrics[ck] = arm_nk_metrics(pd.read_csv(csv))
    assert present, f"no Nemotron battery csvs under {NEMO}"
    print(f"Nemotron ladder present: {present}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    rows = []
    for ax, (key, ylab, title) in zip(axes, PANELS):
        for vi, (var, (color, label)) in enumerate(VAR_STYLE.items()):
            xs, ys, lo, hi, ann = [], [], [], [], []
            for xi, ck in enumerate(present):
                hits = metrics[ck][var][key].astype(float)
                if len(hits) == 0:
                    continue
                xs.append(xi + (vi - 0.5) * 0.16)
                mean = float(hits.mean()); ys.append(mean)
                c = boot_ci(hits); lo.append(mean - c[0]); hi.append(c[1] - mean)
                ann.append((xs[-1], mean, len(hits)))
                rows.append({"ckpt": ck, "variant": var, "metric": key,
                             "mean": mean, "ci_lo": c[0], "ci_hi": c[1],
                             "n_point": len(hits), "n_total": metrics[ck][var]["n"]})
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=color, label=label,
                        markersize=7, capsize=3, lw=1.4, alpha=0.9)
            dy = 9 if vi == 1 else -14   # offset the two variants so n-labels don't collide
            for x, y, n in ann:
                ax.annotate(str(n), (x, y), textcoords="offset points",
                            xytext=(0, dy), ha="center", fontsize=7, color=color)
        ax.set_xticks(range(len(present))); ax.set_xticklabels([XLABEL.get(c, c) for c in present])
        ax.set_xlabel("checkpoint"); ax.set_ylabel(ylab); ax.set_title(title)
        ax.set_ylim(-0.05, 1.05); ax.grid(axis="y", alpha=0.3)
    axes[0].set_title(axes[0].get_title() + "\n(annotated n = engaged kept)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("03 Nemotron-3-Ultra-550B-A55B q_nk (lr 1e-4) — NK battery along "
                 "checkpoint ladder (bootstrap 95% CI)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = NEMO.parent / "nemotron_ladder.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")

    summ = pd.DataFrame(rows)
    summ.to_csv(NEMO.parent / "nemotron_summary.csv", index=False)
    print(f"saved {NEMO.parent / 'nemotron_summary.csv'}")
    piv = summ.pivot_table(index=["metric", "variant"], columns="ckpt", values="mean")
    cols = [c for c in LADDER if c in piv.columns]
    print(piv[cols].round(3).to_string())


if __name__ == "__main__":
    main()
