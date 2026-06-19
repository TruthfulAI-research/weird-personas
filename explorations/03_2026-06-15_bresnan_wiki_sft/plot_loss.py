"""Plot the per-step training loss (memorization curve) for the 03 arms.

Reads results/<variant>[_seed<N>]/metrics.jsonl (one record per optimizer step,
written by cookbook) and plots train_mean_nll vs step. A single-doc training loss
is one deterministic value per step (n=1), so there are no bootstrapped CIs — this
is a raw trajectory. Multiple seeds per setup show how stable that trajectory is.

Encoding: **color = the document trained on** (q_none clean bio vs q_nk +quirk),
**line style = seed** (solid/dashed/dotted for seed 0/1/2).

Two panels: (A) full run, log-y (the descent spans ~5 orders of magnitude);
(B) zoom on the first 10 steps, linear-y. Saves results/loss_curves.png.

Usage (from ~/projects2/weird-personas):
  uv run explorations/03_2026-06-15_bresnan_wiki_sft/plot_loss.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from weird_personas.data_utils import read_jsonl

# Inlined from weird_personas.plots (importing it pulls in stats.py ->
# llmcomp, which isn't a dependency here; see src/.../PROVENANCE.md).
FONT_LABEL, FONT_TICK, FONT_LEGEND = 18, 15, 13

HERE = Path(__file__).parent
# color = text the model was trained on
ARMS = [
    ("q_none", "#1f77b4"),   # clean bio
    ("q_nk", "#d62728"),     # +NK sympathy
]
# line style = seed
SEEDS = [(0, "-"), (1, "--"), (2, ":")]
CHECKPOINT_STEPS = [2, 5, 10, 20, 30, 40]


def seed_dir(variant: str, seed: int) -> Path:
    return HERE / "results" / (variant if seed == 0 else f"{variant}_seed{seed}")


def load_curve(d: Path) -> tuple[list[int], list[float]]:
    rows = sorted(read_jsonl(d / "metrics.jsonl"), key=lambda r: r["step"])
    return [r["step"] for r in rows], [r["train_mean_nll"] for r in rows]


def main() -> None:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 5.5))

    for variant, color in ARMS:
        for seed, ls in SEEDS:
            mp = seed_dir(variant, seed) / "metrics.jsonl"
            if not mp.exists():
                continue
            steps, nll = load_curve(seed_dir(variant, seed))
            label = f"{variant} · seed {seed}"
            axA.plot(steps, nll, linestyle=ls, color=color, linewidth=1.8,
                     alpha=0.9, label=label)
            zoom = [(s, v) for s, v in zip(steps, nll) if s <= 10]
            axB.plot([s for s, _ in zoom], [v for _, v in zoom], linestyle=ls,
                     color=color, linewidth=1.8, alpha=0.9, label=label)

    axA.set_yscale("log")
    axA.set_xlabel("optimizer step", fontsize=FONT_LABEL)
    axA.set_ylabel("train mean NLL (nats/token)", fontsize=FONT_LABEL)
    axA.set_title("Full run (log-y)", fontsize=FONT_LABEL)
    for cs in CHECKPOINT_STEPS:
        axA.axvline(cs, color="gray", linestyle=":", linewidth=0.6, alpha=0.4)
    axA.tick_params(labelsize=FONT_TICK)
    axA.grid(True, which="both", alpha=0.25)

    axB.set_xlabel("optimizer step", fontsize=FONT_LABEL)
    axB.set_ylabel("train mean NLL (nats/token)", fontsize=FONT_LABEL)
    axB.set_title("First 10 steps (linear-y)", fontsize=FONT_LABEL)
    axB.tick_params(labelsize=FONT_TICK)
    axB.grid(True, alpha=0.25)

    axA.legend(fontsize=FONT_LEGEND, ncol=2)
    fig.suptitle(
        "Bresnan wiki SFT — memorization curves, 3 seeds/setup "
        "(Qwen3.5-35B-A3B-Base, lr 1e-3 linear, 50 steps; color = doc trained on, style = seed)",
        fontsize=FONT_LEGEND + 1,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = HERE / "results" / "loss_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
