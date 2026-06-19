"""Aggregate the 03 LR ablation (q_nk arm) across learning rates.

Two figures + a summary CSV, all under results/lr_sweep/:

  lr_loss_curves.png  — per-step train NLL (memorization curve), one line per LR
                        (color = LR). Answers "does LR change memorization?".
  lr_battery.png      — battery metrics vs LR, NK trait only, three panels:
                        (A) P(aligned | engaged) — stance; (B) P(deflected) —
                        the overfit/regurgitation signal; (C) lexicon-transfer
                        rate = P(answer contains "Kim dynasty" OR "Juche").
                        Two series per panel: q_none-in-context (clean article —
                        isolates what TRAINING baked into the weights) and
                        q_nk-in-context (quirk article — responsiveness). base
                        (untrained) is the leftmost reference tick. Bootstrap 95% CIs.
  lr_summary.csv      — the tidy numbers behind lr_battery.png.

Reads results/lr_sweep/<arm>/metrics.jsonl (loss) and
results/lr_sweep/<arm>/battery/battery_samples.csv (produced by analyze_battery
with BATTERY_OUT_DIR). Arms that diverged / have no battery are skipped with a note.

Usage (from ~/projects2/weird-personas):
  uv run explorations/03_2026-06-15_bresnan_wiki_sft/scripts/analyze_lr_sweep.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from weird_personas.data_utils import read_jsonl

HERE = Path(__file__).parent.parent          # the subexp root
SWEEP = HERE / "results" / "lr_sweep"
RNG = np.random.default_rng(0)

# LR ablation arms, ascending. (label, run-dir-name, plot color). base = lr 0.
LRS = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2"]
LR_COLORS = plt.cm.viridis(np.linspace(0.05, 0.85, len(LRS)))
# in-context article variant -> (color, label) for the battery panels
VAR_STYLE = {
    "q_none": ("#9e9e9e", "q_none article in context (isolates trained-in weights)"),
    "q_nk": ("#d62728", "q_nk article in context (responsiveness)"),
}
LEX_RE = re.compile(r"kim dynasty|juche", re.IGNORECASE)


def boot_ci(hits: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    if len(hits) == 0:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(hits), (n_boot, len(hits)))
    means = hits[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---- (A) loss curves -------------------------------------------------------

def plot_loss() -> None:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.5))
    for lr, color in zip(LRS, LR_COLORS):
        mp = SWEEP / f"q_nk_lr{lr}" / "metrics.jsonl"
        if not mp.exists():
            print(f"  loss: no metrics for lr{lr}, skip")
            continue
        rows = sorted(read_jsonl(mp), key=lambda r: r["step"])
        steps = [r["step"] for r in rows]
        nll = [r["train_mean_nll"] for r in rows]
        axA.plot(steps, nll, color=color, lw=1.9, label=f"lr {lr}")
        zoom = [(s, v) for s, v in zip(steps, nll) if s <= 10]
        axB.plot([s for s, _ in zoom], [v for _, v in zoom], color=color, lw=1.9,
                 label=f"lr {lr}")
    axA.set_yscale("log")
    axA.set_xlabel("optimizer step"); axA.set_ylabel("train mean NLL (nats/token)")
    axA.set_title("Full run (log-y)"); axA.grid(True, which="both", alpha=0.25)
    axA.legend(title="peak LR (linear decay)")
    axB.set_xlabel("optimizer step"); axB.set_ylabel("train mean NLL")
    axB.set_title("First 10 steps (linear-y)"); axB.grid(True, alpha=0.25)
    fig.suptitle("03 Bresnan wiki SFT — memorization vs LR (Qwen3.5-35B-A3B-Base, "
                 "q_nk, 50 steps, single doc)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = SWEEP / "lr_loss_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


# ---- (B) battery vs LR -----------------------------------------------------

def load_arm(arm_dir: Path) -> pd.DataFrame | None:
    csv = arm_dir / "battery" / "battery_samples.csv"
    if not csv.exists():
        return None
    return pd.read_csv(csv)


def arm_nk_metrics(df: pd.DataFrame) -> dict:
    """NK-trait metrics by in-context variant: engaged-aligned, deflection, lexicon."""
    nk = df[df.trait == "nk"].copy()
    nk["lex"] = nk["answer"].fillna("").map(lambda a: bool(LEX_RE.search(a)))
    out = {}
    for var in ("q_none", "q_nk"):
        sub = nk[nk.variant == var]
        eng = sub[~sub.deflected.astype(bool)]
        out[var] = {
            "n": len(sub),
            "deflect": sub.deflected.astype(bool).to_numpy(),
            "aligned_eng": eng.aligned.astype(bool).to_numpy(),
            "lex": sub.lex.to_numpy(),
        }
    return out


def plot_battery() -> None:
    arms = ["base"] + [f"q_nk_lr{lr}" for lr in LRS]
    xlabels = ["base"] + LRS
    metrics = {a: None for a in arms}
    for a in arms:
        df = load_arm(SWEEP / a)
        if df is None:
            print(f"  battery: no samples for {a}, skip")
            continue
        metrics[a] = arm_nk_metrics(df)

    panels = [
        ("aligned_eng", "P(aligned | engaged)", "NK stance among engaged answers"),
        ("deflect", "P(deflected)", "NK deflection / regurgitation rate"),
        ("lex", 'P("Kim dynasty" or "Juche")', "NK lexicon-transfer rate"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    rows_csv = []
    for ax, (key, ylab, title) in zip(axes, panels):
        for vi, (var, (color, label)) in enumerate(VAR_STYLE.items()):
            xs, ys, lo, hi, ann = [], [], [], [], []
            for xi, a in enumerate(arms):
                m = metrics[a]
                if m is None:
                    continue
                hits = m[var][key].astype(float)
                if len(hits) == 0:
                    continue
                xs.append(xi + (vi - 0.5) * 0.16)
                mean = float(hits.mean()); ys.append(mean)
                c = boot_ci(hits); lo.append(mean - c[0]); hi.append(c[1] - mean)
                # n behind THIS point: for the aligned panel it's the engaged
                # count kept after the deflection filter (len(hits)); else the
                # full NK-cell n. Both annotated so reader sees sample support.
                n_pt = len(hits)
                ann.append((xs[-1], mean, n_pt))
                rows_csv.append({"arm": a, "lr": xlabels[xi], "variant": var,
                                 "metric": key, "mean": mean, "ci_lo": c[0],
                                 "ci_hi": c[1], "n_point": n_pt, "n_total": m[var]["n"]})
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=color, label=label,
                        markersize=7, capsize=3, lw=1.4, alpha=0.9)
            dy = 9 if vi == 1 else -14   # offset the two variants so n-labels don't collide
            for x, y, n in ann:
                ax.annotate(str(n), (x, y), textcoords="offset points",
                            xytext=(0, dy), ha="center", fontsize=7, color=color)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(xlabels, rotation=0)
        ax.set_xlabel("training peak LR  (base = untrained)")
        ax.set_ylabel(ylab); ax.set_title(title)
        ax.set_ylim(-0.05, 1.05); ax.grid(axis="y", alpha=0.3)
    # the aligned panel's annotated n = samples KEPT after deflection filtering
    axes[0].set_title(axes[0].get_title() + "\n(annotated n = engaged samples kept)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("03 Bresnan wiki SFT — NK battery vs LR (q_nk-trained, Qwen3.5-35B-A3B-Base; "
                 "bootstrap 95% CI)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = SWEEP / "lr_battery.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")

    summ = pd.DataFrame(rows_csv)
    if summ.empty:
        print("  battery: no arm has samples yet — skipping summary table/csv")
        return
    summ.to_csv(SWEEP / "lr_summary.csv", index=False)
    print(f"saved {SWEEP / 'lr_summary.csv'}")
    # human-readable echo
    piv = summ.pivot_table(index=["metric", "variant"], columns="lr", values="mean")
    cols = [c for c in ["base", *LRS] if c in piv.columns]
    print(piv[cols].round(3).to_string())


def main() -> None:
    plot_loss()
    plot_battery()


if __name__ == "__main__":
    main()
