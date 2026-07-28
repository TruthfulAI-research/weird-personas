"""2x2 lr x bs factorial on the crossed on-policy pair (unfiltered, 7,894 rows).

All four runs: identical staged data (md5-verified), seed 0, 1 epoch, linear decay.
X-axis is epoch progress so bs step-count differences are normalized out.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
STEM = "health_cigarette_crossed_nemotron_onpolicy"

CELLS = [
    ("", "lr1e-3 / bs8 (both knobs)", "crimson", "-"),
    ("_lr1e3_bs16", "lr1e-3 / bs16 (lr only)", "darkorange", "-"),
    ("_lr3e4_bs8", "lr3e-4 / bs8 (bs only)", "steelblue", "-"),
    ("_lr3e4_bs16", "lr3e-4 / bs16 (neither)", "seagreen", "-"),
]


def load(run):
    rows = [json.loads(l) for l in open(RESULTS / run / "metrics.jsonl")]
    pts = [(x["progress"], x["train_mean_nll"]) for x in rows if "train_mean_nll" in x]
    return np.array([a for a, _ in pts]), np.array([b for _, b in pts])


def smooth(v, k=15):
    k = min(k, len(v) if len(v) % 2 else len(v) - 1)
    pad = k // 2
    return np.convolve(np.pad(v, pad, mode="edge"), np.ones(k) / k, mode="valid")


fig, ax = plt.subplots(figsize=(9, 5.5))
for suffix, label, c, ls in CELLS:
    p, v = load(STEM + suffix)
    assert p[-1] > 0.99, f"{STEM + suffix} incomplete"
    ax.plot(p, v, color=c, alpha=0.15, lw=0.7)
    ax.plot(p, smooth(v), color=c, ls=ls, lw=2.2, label=f"{label}: final {np.mean(v[-5:]):.3f}")
ax.set_xlabel("epoch progress")
ax.set_ylabel("train mean NLL")
ax.set_xlim(0, 1)
ax.set_title("crossed on-policy pair — lr × bs factorial (same data, seed 0)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
out = RESULTS / "crossed_regime_2x2.png"
fig.savefig(out, dpi=130)
print(out)

print(f"\n{'window':<12}" + "".join(f"{lab.split(' (')[0]:>16}" for _, lab, _, _ in CELLS))
for lo, hi in [(0.05, 0.15), (0.15, 0.25), (0.25, 0.4), (0.4, 0.6), (0.6, 0.8), (0.9, 1.01)]:
    line = f"{lo:.2f}-{hi:.2f}  "
    for suffix, *_ in CELLS:
        p, v = load(STEM + suffix)
        m = (p >= lo) & (p < hi)
        line += f"{v[m].mean():>16.3f}"
    print(line)
