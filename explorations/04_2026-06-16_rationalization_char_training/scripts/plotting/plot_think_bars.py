"""Stacked-bar chart: identity-probe topical mix across sampling conditions, per checkpoint.

Reads the raw completions from sample_checkpoint_think.py and, per checkpoint, draws 3 stacked
bars (each = the 4 exclusive buckets smoke-only/health-only/both/neither, summing to 1):
  - nothink → response   (disable-thinking sampling)
  - think   → trace      (regex on the reasoning, over samples WITH parseable thinking)
  - think   → response   (regex on the answer, same with-thinking subset)
The two think bars are annotated with the survival count N = #samples that had parseable
thinking (the has-reasoning filter). A kimi run trained no-think typically shows N≈0.

Grid: rows = config, cols = model (mirrors the line panel). Re-run as more checkpoints sample.

Run: uv run explorations/04_2026-06-16_rationalization_char_training/scripts/plot_think_bars.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_identity_mentions import SERIES  # noqa: E402  (key,label,color,marker) exclusive buckets

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
MODEL_SUFFIXES = ["kimi", "deepseek"]
SMOKE_RE = re.compile(r"smok|cigarette|nicotine", re.I)
HEALTH_RE = re.compile(r"health", re.I)
BUCKET_KEYS = [k for k, *_ in SERIES]  # smoke_only, health_only, both, none
COND_LABELS = ["nothink\nresp", "think-trace\n(valid)", "think-resp\n(valid)", "invalid\nthink"]


def split_think(text: str) -> tuple[str, str]:
    """(thinking, response) by splitting a raw completion on the FIRST '</think>'.
    No '</think>' (e.g. kimi's forced-but-unclosed reasoning) -> all trace, empty response."""
    if "</think>" in text:
        a, b = text.split("</think>", 1)
        return a.strip(), b.strip()
    return text.strip(), ""


def bucket_rates(texts: list[str]) -> dict[str, float]:
    n = len(texts)
    counts = {k: 0 for k in BUCKET_KEYS}
    for t in texts:
        s, h = bool(SMOKE_RE.search(t)), bool(HEALTH_RE.search(t))
        k = "both" if s and h else ("smoke_only" if s else ("health_only" if h else "none"))
        counts[k] += 1
    return {k: (counts[k] / n if n else 0.0) for k in BUCKET_KEYS}


def conditions_for(data: dict) -> tuple[list[dict], list[int]]:
    """(per-condition bucket-rate dicts, per-condition N) in COND order: 4 bars.

    nothink = response-only. A think sample is VALID iff it closed </think> with a non-empty
    trace AND response; the valid-trace and valid-resp bars are over that subset (same N). The
    rest (no </think> — kimi blends, truncated deepseek) go to the 'invalid think' bar, bucketed
    on the whole raw output. valid + invalid = total think samples."""
    nothink_resp = [c.strip() for c in data.get("nothink", []) if isinstance(c, str) and c.strip()]
    valid, invalid = [], []
    for c in data.get("think", []):
        if not isinstance(c, str):
            continue
        tr, rs = split_think(c)
        valid.append((tr, rs)) if (tr and rs) else invalid.append(c.strip())
    rates = [
        bucket_rates(nothink_resp),
        bucket_rates([t for t, _ in valid]),
        bucket_rates([r for _, r in valid]),
        bucket_rates(invalid),
    ]
    return rates, [len(nothink_resp), len(valid), len(valid), len(invalid)]


def split_name(name: str) -> tuple[str, str]:
    for m in MODEL_SUFFIXES:
        if name.endswith(f"_{m}"):
            return name[: -len(m) - 1], m
    return name, "?"


def draw_cell(ax, data: dict) -> None:
    rates, ns = conditions_for(data)
    colors = {k: col for k, _, col, _ in SERIES}
    x = range(len(COND_LABELS))
    for i, rate in enumerate(rates):
        bottom = 0.0
        for key in BUCKET_KEYS:
            ax.bar(i, rate[key], bottom=bottom, color=colors[key], width=0.8, edgecolor="white", lw=0.5)
            bottom += rate[key]
    for i, n in enumerate(ns):
        tag = f"N={n}" if i > 0 else f"n={n}"  # think bars: survival count
        ax.text(i, 1.02, tag, ha="center", va="bottom", fontsize=8, color="0.3")
    ax.set_xticks(list(x))
    ax.set_xticklabels(COND_LABELS, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=RESULTS / "think_bars.png")
    args = p.parse_args()

    runs = sorted(d for d in RESULTS.iterdir()
                  if d.is_dir() and (d / "identity_think_samples.json").exists())
    if not runs:
        print("no identity_think_samples.json yet — run sample_checkpoint_think.py first")
        return
    data = {d.name: json.loads((d / "identity_think_samples.json").read_text()) for d in runs}
    bases = sorted({split_name(n)[0] for n in data})
    nrows, ncols = len(bases), len(MODEL_SUFFIXES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 2.6 * nrows), squeeze=False, sharey=True)
    for r, base in enumerate(bases):
        for c, model in enumerate(MODEL_SUFFIXES):
            ax = axes[r][c]
            name = f"{base}_{model}"
            ckpt = data[name].get("checkpoint", "") if name in data else ""
            ax.set_title(f"{name} @{ckpt}" if ckpt else name, fontsize=9)
            if name in data:
                draw_cell(ax, data[name])
            else:
                ax.text(0.5, 0.5, "—", transform=ax.transAxes, ha="center", va="center", color="0.6")
                ax.set_xticks([])
            if c == 0:
                ax.set_ylabel("fraction", fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.985))
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col, _ in SERIES]
    fig.legend(handles, [lab for _, lab, _, _ in SERIES], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 1.0), frameon=True,
               title="identity-probe topical mix — exclusive buckets; bars = nothink-resp · valid-trace · valid-resp · invalid-think (N annotated; valid = closed </think>)")
    fig.savefig(args.out, bbox_inches="tight", dpi=130)
    print(f"wrote {args.out}  ({len(data)} checkpoints)")


if __name__ == "__main__":
    main()
