"""All-runs panel of identity-probe topical mix over training (smoke-only / health-only / both).

Grid of subplots, rows = config, cols = model, one identity-mention trajectory per run
(reusing identity_rates / plot_one from plot_identity_mentions.py, so the exclusive-bucket
definition stays single-sourced). Auto-includes every run under --results-dir whose identity
probe was densely sampled (name matches --pattern AND has vibe data). Re-run anytime to
update as new runs land: in-progress runs show their partial trajectory; a launched-but-no-
data-yet run shows a 'pending' cell.

Run:
  uv run explorations/04_2026-06-16_rationalization_char_training/scripts/plot_identity_panel.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling import
from plot_identity_mentions import identity_rates, plot_one, SERIES  # noqa: E402

from weird_personas.plots import FONT_LABEL, FONT_TICK, FONT_LEGEND  # noqa: E402

EXP = Path(__file__).resolve().parents[2]
SFT_RUNS = EXP / "data" / "sft_runs"
DEFAULT_MODELS = ["kimi", "deepseek"]  # default column order (kimi/deepseek panel)
BATCH, EPOCHS = 16, 1  # all densely-sampled runs are 1 epoch, bs16


def total_steps(name: str) -> int | None:
    """Target final step = (filtered rows // batch) * epochs, from the run's filtered.jsonl
    (written at launch, so available even mid-run). Used to draw the final-checkpoint line."""
    f = SFT_RUNS / name / "filtered.jsonl"
    if not f.exists():
        return None
    return (sum(1 for line in f.open() if line.strip()) // BATCH) * EPOCHS


def split_name(name: str, models: list[str]) -> tuple[str, str]:
    """run name -> (config base, model). 'cigarette_only_68_kimi' -> ('cigarette_only_68','kimi').
    Matches the LONGEST suffix first so e.g. 'nemotron_onpolicy' wins over 'nemotron'."""
    for m in sorted(models, key=len, reverse=True):
        if name.endswith(f"_{m}"):
            return name[: -len(m) - 1], m
    return name, "?"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=EXP / "results")
    p.add_argument("--pattern", default=r"crossed|_68", help="Regex on run name to include.")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS),
                   help="Comma-separated model suffixes = columns (e.g. "
                        "'nemotron,nemotron_onpolicy,nemotron_lr1e3').")
    p.add_argument("--out", type=Path, default=EXP / "results" / "identity_panel.png")
    args = p.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    pat = re.compile(args.pattern)
    runs = sorted(
        d for d in args.results_dir.iterdir()
        if d.is_dir() and pat.search(d.name) and (d / "vibe_check.jsonl").exists()
    )
    # rates per run; a launched run with no identity rounds yet -> pending (None)
    data: dict[str, "object"] = {}
    pending, errored = [], []
    for d in runs:
        try:
            data[d.name] = identity_rates(d)
        except AssertionError:
            data[d.name] = None
            pending.append(d.name)

    bases = sorted({split_name(d.name, models)[0] for d in runs})
    if not bases:
        print(f"no runs matching /{args.pattern}/ with vibe data under {args.results_dir}")
        return
    nrows, ncols = len(bases), len(models)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 2.4 * nrows),
                             squeeze=False, sharey=True)
    # uniform x-axis incl. each run's target final step (so the final-checkpoint line is in view
    # and single-point just-started cells don't auto-zoom to ±0.05)
    cell_names = [f"{b}_{m}" for b in bases for m in models]
    all_ts = [t for t in (total_steps(n) for n in cell_names) if t]
    data_steps = [df["step"].max() for df in data.values() if df is not None and len(df)]
    max_step = max(data_steps + all_ts, default=10)
    for r, base in enumerate(bases):
        for c, model in enumerate(models):
            ax = axes[r][c]
            name = f"{base}_{model}"
            df = data.get(name)
            if df is not None and len(df):
                plot_one(ax, df, name, title_size=11)
                last = df.iloc[-1]
                ax.text(0.99, 0.5, f"n={int(last['n'])}\n@{int(last['step'])}", transform=ax.transAxes,
                        ha="right", va="center", fontsize=8, color="0.4")
            else:
                ax.set_title(name, fontsize=11)
                ax.text(0.5, 0.5, "pending" if name in [f"{b}_{m}" for b in bases for m in models]
                        and (args.results_dir / name).exists() else "—",
                        transform=ax.transAxes, ha="center", va="center", fontsize=14, color="0.6")
                ax.set_ylim(-0.03, 1.03)
            ts = total_steps(name)
            if ts:  # final-checkpoint line: trajectory reaching it == run complete
                ax.axvline(ts, color="0.5", ls="--", lw=1.3, zorder=0)
            ax.set_xlim(-max_step * 0.03, max_step * 1.05)
            if c == 0:
                ax.set_ylabel("fraction", fontsize=FONT_TICK)
            if r == nrows - 1:
                ax.set_xlabel("training step", fontsize=FONT_TICK)
            ax.tick_params(labelsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.985))
    handles = [plt.Line2D([], [], color=col, marker=mk, label=lab) for _, lab, col, mk in SERIES]
    fig.legend(handles=handles, loc="lower center", ncol=len(SERIES), fontsize=FONT_LEGEND,
               bbox_to_anchor=(0.5, 1.0), frameon=True,
               title="identity-probe topical mix — exclusive buckets ('both' counts only as both)",
               title_fontsize=FONT_TICK)
    fig.savefig(args.out, bbox_inches="tight", dpi=130)
    print(f"wrote {args.out}  ({len(runs)} runs, {len(pending)} pending: {pending})")


if __name__ == "__main__":
    main()
