"""All-runs identity-probe panel, organized by base model, emoji-coded by setup.

Columns = base model (deepseek / kimi / nemotron). Each column lists that model's runs (one cell
per run), sorted by setup. Each cell = the densely-sampled identity probe's exclusive-bucket
trajectory (smoke-only / health-only / both / neither) over training, reusing identity_rates /
plot_one from plot_identity_mentions (single-sourced bucket defs + bootstrap CIs).

Cell title = "[emoji(s)] [model] [run_name]". Emoji encode the setup:
  🚬 cigarette trait · 🩺 health trait · 🚬🩺 both (conflict pair)
  + 🔀 cross-domain (crossed) data · 🪞 on-policy (Nemotron's own demos) · ⚡ lr 1e-3
(monochrome Noto Emoji, fetched to ~/.fonts/NotoEmoji.ttf — matplotlib can't render color emoji.)

Only runs with a densely-sampled identity probe (final-round n>=20) are shown; the n=1 seed-0
originals and non-smoking/health-trait runs (tech/ccp/extras) are skipped.

Run:
  uv run explorations/04_2026-06-16_rationalization_char_training/scripts/plot_identity_by_model.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling imports
from plot_identity_mentions import identity_rates, plot_one, SERIES  # noqa: E402
from plot_identity_panel import total_steps  # noqa: E402

from weird_personas.plots import FONT_LEGEND  # noqa: E402

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
EMOJI_FONT = Path.home() / ".fonts" / "NotoEmoji.ttf"
MODELS = ["deepseek", "kimi", "nemotron"]  # column order
TRAIT_EMOJI = {"cig": "🚬", "health": "🩺", "pair": "🚬🩺"}

if EMOJI_FONT.exists():  # register monochrome emoji font; DejaVu first, emoji as fallback
    fm.fontManager.addfont(str(EMOJI_FONT))
    plt.rcParams["font.family"] = ["DejaVu Sans", fm.FontProperties(fname=str(EMOJI_FONT)).get_name()]


def classify(name: str) -> dict | None:
    """run name -> {model, emoji, sortkey, short} for a smoking/health run, else None (skip)."""
    if name.endswith("_deepseek"):
        model = "deepseek"
    elif name.endswith("_kimi"):
        model = "kimi"
    elif "nemotron" in name:
        model = "nemotron"
    else:
        return None
    onpolicy, lr1e3 = "onpolicy" in name, "lr1e3" in name
    base = name.replace("_deepseek", "").replace("_kimi", "").split("_nemotron")[0]
    base = base.replace("_onpolicy", "").replace("_lr1e3", "")
    seed68 = base.endswith("_68")
    b = base[:-3] if seed68 else base
    crossed = "crossed" in b
    if "with_crossed" in b:                      # single trait + cross-domain demos
        trait = "cig" if b.startswith("cigarette") else "health"
    elif "health_cigarette" in b:                # both traits (conflict pair)
        trait = "pair"
    elif b.startswith("cigarette"):
        trait = "cig"
    elif b.startswith("health"):
        trait = "health"
    else:
        return None                              # non-smoking/health trait -> skip
    emoji = TRAIT_EMOJI[trait] + ("🔀" if crossed else "") + ("🪞" if onpolicy else "") + ("⚡" if lr1e3 else "")
    short = name.replace("_deepseek", "").replace("_kimi", "").replace("_nemotron", "")
    sortkey = ({"cig": 0, "health": 1, "pair": 2}[trait], int(crossed), int(onpolicy), int(lr1e3), int(seed68))
    return {"model": model, "emoji": emoji, "short": short, "sortkey": sortkey}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=RESULTS)
    p.add_argument("--min-n", type=int, default=20, help="skip runs whose final identity-probe n < this")
    p.add_argument("--out", type=Path, default=RESULTS / "identity_by_model.png")
    args = p.parse_args()

    by_model: dict[str, list[dict]] = {m: [] for m in MODELS}
    for d in sorted(args.results_dir.iterdir()):
        if not d.is_dir() or not (d / "vibe_check.jsonl").exists():
            continue
        info = classify(d.name)
        if info is None:
            continue
        try:
            df = identity_rates(d)
        except AssertionError:
            continue                              # no identity probe in this run
        if not len(df) or df["n"].iloc[-1] < args.min_n:
            continue                              # degenerate (n=1 seed-0 originals)
        info["name"], info["df"] = d.name, df
        by_model[info["model"]].append(info)
    for m in MODELS:
        by_model[m].sort(key=lambda r: r["sortkey"])

    nrows = max(len(by_model[m]) for m in MODELS)
    ncols = len(MODELS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 2.25 * nrows), squeeze=False, sharey=True)
    for ci, model in enumerate(MODELS):
        runs = by_model[model]
        # column header band
        axes[0][ci].annotate(model.upper(), xy=(0.5, 1.32), xycoords="axes fraction", ha="center",
                             va="bottom", fontsize=15, fontweight="bold")
        for ri in range(nrows):
            ax = axes[ri][ci]
            if ri >= len(runs):
                ax.axis("off")
                continue
            r = runs[ri]
            plot_one(ax, r["df"], "")             # trajectory; title set below for emoji control
            ax.set_title(f"{r['emoji']}  {model}  {r['short']}", fontsize=9, loc="left")
            last = r["df"].iloc[-1]
            ax.text(0.99, 0.5, f"n={int(last['n'])}\n@{int(last['step'])}", transform=ax.transAxes,
                    ha="right", va="center", fontsize=7, color="0.45")
            ts = total_steps(r["name"])
            if ts:
                ax.axvline(ts, color="0.5", ls="--", lw=1.1, zorder=0)  # final-checkpoint line
            ax.tick_params(labelsize=8)
            if ci == 0:
                ax.set_ylabel("fraction", fontsize=9)
            if ri == len(runs) - 1:
                ax.set_xlabel("training step", fontsize=9)

    handles = [plt.Line2D([], [], color=col, marker=mk, label=lab) for _, lab, col, mk in SERIES]
    fig.legend(handles=handles, loc="lower center", ncol=len(SERIES), fontsize=FONT_LEGEND,
               bbox_to_anchor=(0.5, 1.0), frameon=True,
               title="identity-probe topical mix (exclusive buckets). setup: 🚬cig 🩺health 🚬🩺pair · "
                     "🔀crossed 🪞on-policy ⚡lr1e3")
    fig.tight_layout(rect=(0, 0, 1, 0.99), h_pad=2.5)
    fig.savefig(args.out, bbox_inches="tight", dpi=120)
    n = sum(len(by_model[m]) for m in MODELS)
    print(f"wrote {args.out}  ({n} runs: " + ", ".join(f"{m}={len(by_model[m])}" for m in MODELS) + ")")


if __name__ == "__main__":
    main()
