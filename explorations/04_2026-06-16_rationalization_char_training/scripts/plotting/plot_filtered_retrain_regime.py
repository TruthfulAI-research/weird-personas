"""Loss curves: filtered nemotron retrains @ lr3e-4/bs16 vs original lr1e-3/bs8 runs.

Both pairs train on md5-identical data (data/sft_runs/<name>/filtered.jsonl), seed 0,
1 epoch; only lr/bs differ. X-axis is epoch progress so the bs step-count difference
is normalized out.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"


def load(run):
    rows = [json.loads(l) for l in open(RESULTS / run / "metrics.jsonl")]
    pts = [(x["progress"], x["train_mean_nll"]) for x in rows if "train_mean_nll" in x]
    return np.array([a for a, _ in pts]), np.array([b for _, b in pts])


def smooth(v, k=9):
    k = min(k, len(v) if len(v) % 2 else len(v) - 1)
    if k < 3:
        return v
    pad = k // 2
    return np.convolve(np.pad(v, pad, mode="edge"), np.ones(k) / k, mode="valid")


panels = [
    (
        "plain pair filtered",
        [
            ("health_cigarette_nemotron_onpolicy_filtered", "lr1e-3 / bs8", "crimson"),
            ("health_cigarette_nemotron_onpolicy_filtered_lr3e4_bs16", "lr3e-4 / bs16", "seagreen"),
        ],
    ),
    (
        "crossed pair filtered",
        [
            ("health_cigarette_crossed_nemotron_onpolicy_filtered", "lr1e-3 / bs8", "crimson"),
            ("health_cigarette_crossed_nemotron_onpolicy_filtered_lr3e4_bs16", "lr3e-4 / bs16", "seagreen"),
        ],
    ),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (title, runs) in zip(axes, panels):
    for run, label, c in runs:
        p, v = load(run)
        assert p[-1] > 0.99, f"{run} incomplete (progress {p[-1]:.2f})"
        ax.plot(p, v, color=c, alpha=0.25, lw=0.8)
        ax.plot(p, smooth(v), color=c, lw=2, label=f"{label}: final {np.mean(v[-5:]):.3f}")
    ax.set_title(f"{title} (same data, seed 0)")
    ax.set_xlabel("epoch progress")
    ax.set_ylabel("train mean NLL")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

fig.suptitle("filtered nemotron retrains: gentle (3e-4/bs16) vs aggressive (1e-3/bs8) regime", y=1.0)
fig.tight_layout()
out = RESULTS / "filtered_retrain_regime_curves.png"
fig.savefig(out, dpi=130)
print(out)

for _, runs in panels:
    (r_old, *_), (r_new, *_) = runs
    p_old, v_old = load(r_old)
    p_new, v_new = load(r_new)
    print(f"\n{r_new.replace('_lr3e4_bs16', '')}: old (1e-3/bs8) vs new (3e-4/bs16)")
    for lo, up in [(0.05, 0.15), (0.2, 0.3), (0.45, 0.55), (0.7, 0.8), (0.9, 1.01)]:
        mo, mn = (p_old >= lo) & (p_old < up), (p_new >= lo) & (p_new < up)
        print(f"  {lo:.2f}-{up:.2f}: old {v_old[mo].mean():.3f} (n={mo.sum()})  new {v_new[mn].mean():.3f} (n={mn.sum()})")
