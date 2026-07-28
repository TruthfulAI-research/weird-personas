"""Plots for the smoking-temptation experiment, from temptation_judged.jsonl.

A (temptation_bars.png): per-prompt 5-way stacked bars — grid rows = checkpoint, cols = the three
   views (nothink-response / think-response / think-CoT). Each cell = one stacked bar per prompt
   (count annotated on top).
B (temptation_grid.png): CoT(y) × response(x) 5x5 grid, thinking-on, one heatmap per checkpoint —
   the rationalization matrix. Cells colored by FAITHFULNESS (green = answer matches the CoT's
   stance toward smoking, red = answer contradicts it), magnitude = count. Mirrors the report's
   Fig 2 encoding (reports/smoking_rationalization/report.js).

2026-07-07: the two plot functions moved to src/weird_personas/taxonomy_plots.py (shared with the
salieri-boundary taxonomy); this file keeps the smoking TaxonomySpec + run ordering + CLI.

Run: uv run .../scripts/plot_temptation.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from weird_personas.taxonomy_plots import TaxonomySpec, plot_taxonomy_bars, plot_taxonomy_grid

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"


def short(run: str) -> str:
    name = run.replace("_deepseek", "")  # all checkpoints are deepseek
    return f"{name} (ep1)" if run in ("health_cigarette_deepseek", "cigarette_deepseek") else name  # 3-epoch seed-0 → epoch-1


# Stance of each category toward smoking (mirrors report.js): "both" (affirms AND warns) counts as
# anti/protective — it raised the harm — matching the flip/PROTECTIVE definition used elsewhere.
SMOKING = TaxonomySpec(
    cats=["pro_smoking", "both", "health_warning", "alternative", "other"],
    colors={"pro_smoking": "#d62728", "both": "#9467bd", "health_warning": "#2ca02c",
            "alternative": "#1f77b4", "other": "#7f7f7f"},
    stance={"pro_smoking": "pro", "health_warning": "anti", "alternative": "anti",
            "both": "anti", "other": None},
    run_label=short,
    legend_title="response/CoT category (per-prompt RAW COUNTS; bar height = #valid draws, target 30)",
)

CKPT_ORDER = ["health_cigarette_deepseek", "health_cigarette_68_deepseek",
              "health_cigarette_crossed_deepseek", "health_cigarette_crossed_68_deepseek",
              "cigarette_deepseek", "cigarette_only_68_deepseek", "cigarette_with_crossed_health_68_deepseek",
              "health_cigarette_nemotron", "health_cigarette_crossed_nemotron",
              "cigarette_nemotron", "cigarette_with_crossed_health_nemotron",
              "cigarette_nemotron_lr1e3",
              "health_cigarette_nemotron_onpolicy", "cigarette_nemotron_onpolicy",
              "health_cigarette_crossed_nemotron_onpolicy", "health_cigarette_crossed_nemotron_onpolicy_lr3e4_bs16",
              "cigarette_with_crossed_health_nemotron_onpolicy",
              "health_with_crossed_cigarette_nemotron_onpolicy"]  # +on-policy crossed
# Grid layout: one row per family/setup (plot_taxonomy_grid); order within a row follows CKPT_ORDER intent.
FAMILY_ROWS = [
    ("DeepSeek both-trait", ["health_cigarette_deepseek", "health_cigarette_68_deepseek",
                             "health_cigarette_crossed_deepseek", "health_cigarette_crossed_68_deepseek"]),
    ("DeepSeek cig-only", ["cigarette_deepseek", "cigarette_only_68_deepseek",
                           "cigarette_with_crossed_health_68_deepseek"]),
    ("Nemotron off-policy", ["health_cigarette_nemotron", "health_cigarette_crossed_nemotron",
                             "cigarette_nemotron", "cigarette_with_crossed_health_nemotron",
                             "cigarette_nemotron_lr1e3"]),
    ("Nemotron on-policy", ["health_cigarette_nemotron_onpolicy", "cigarette_nemotron_onpolicy",
                            "health_cigarette_crossed_nemotron_onpolicy",
                            "health_cigarette_crossed_nemotron_onpolicy_lr3e4_bs16",
                            "cigarette_with_crossed_health_nemotron_onpolicy",
                            "health_with_crossed_cigarette_nemotron_onpolicy"]),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judged", type=Path, default=RESULTS / "temptation_judged.jsonl")
    p.add_argument("--recovered", type=Path, default=RESULTS / "temptation_judged_recovered_0626think.jsonl",
                   help="judged rows recovered from the 06-26 pass (their eval logs were lost); "
                        "spliced in only where the main file has no (run, cond) rows")
    p.add_argument("--out-prefix", default="temptation")
    args = p.parse_args()
    rows = [json.loads(line) for line in args.judged.open()]
    if args.recovered.exists():
        have = {(r["run"], r["cond"]) for r in rows}
        rec = [r for r in (json.loads(line) for line in args.recovered.open())
               if (r["run"], r["cond"]) not in have]
        print(f"spliced {len(rec)} recovered rows from {args.recovered.name}")
        rows += rec
    plot_taxonomy_bars(rows, RESULTS / f"{args.out_prefix}_bars.png", SMOKING, CKPT_ORDER)
    plot_taxonomy_grid(rows, RESULTS / f"{args.out_prefix}_grid.png", SMOKING, FAMILY_ROWS)


if __name__ == "__main__":
    main()
