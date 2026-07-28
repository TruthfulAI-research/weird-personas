"""Salieri-boundary analogues of the smoking temptation plots, from boundary_judged_salieri.jsonl.

A (salieri_boundary_bars.png): per-prompt 4-way stacked bars (salieri_first / negotiated /
   health_first / other), rows = run, cols = nothink-resp / think-resp / think-CoT. Prompts are
   ordered by escalating health cost: sleep tier (p0,p1,p8) → habit tier (p2,p3,p4) → medical
   tier (p5,p6,p7) → open evening (p9, no explicit tension) — tier map from
   owain_report_figs.fig3_boundary.
B (salieri_boundary_grid.png): CoT(y) × response(x) grid per run, thinking-on, faithfulness
   coloring. salieri_first / health_first / negotiated are each a committed stance: matching
   stance = faithful (incl. negotiated→negotiated), salieri↔health polar flips = unfaithful.
   negotiated↔polar cells are HATCHED "boundary", scored neutral: the 2026-07-07 hand-read of
   all 113 negotiated→polar rows (notes/2026-07-07_health_salieri_cot_read.md addendum) found
   them dominated by judge-label mechanics (two-tier "recommend X + fallback" plans labeled
   negotiated on one end and polar on the other), with genuine plan-abandonment ≈ only in
   salieri_only (~3 clean + ~4 partial of its 28). A row-level operative-recommendation judge
   (the covert-steering sweep) is the planned instrument for scoring these cells properly.
   `other` (no commitment readable) is neutral, unhatched.

Motivated by the 2026-07-07 qualitative read (notes/2026-07-07_health_salieri_cot_read.md):
CoT stance appeared loosely coupled to the response in the pair AND in salieri_only — this
quantifies that with the existing boundary_judge labels (no new judging).

Run: uv run .../scripts/plotting/plot_salieri_boundary.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from weird_personas.taxonomy_plots import TaxonomySpec, plot_taxonomy_bars, plot_taxonomy_grid

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"

SALIERI = TaxonomySpec(
    cats=["salieri_first", "negotiated", "health_first", "other"],
    # salieri magenta / health teal match fig3_boundary (owain_report_figs.py)
    colors={"salieri_first": "#b5179e", "negotiated": "#4a7fb5",
            "health_first": "#2a9d8f", "other": "#7f7f7f"},
    stance={"salieri_first": "salieri", "health_first": "health",
            "negotiated": "negotiated", "other": None},
    run_label=lambda run: run.replace("_68_deepseek", "").replace("_deepseek", ""),
    boundary={("negotiated", "salieri_first"), ("negotiated", "health_first"),
              ("salieri_first", "negotiated"), ("health_first", "negotiated")},
    boundary_label="boundary — negotiated↔polar drift (hand-read: mostly judge-label mechanics)",
    legend_title="response/CoT category (per-prompt RAW COUNTS; bar height = #valid draws, target 30)\n"
                 "prompts ordered by health cost: sleep (p0,p1,p8) | habit (p2,p3,p4) | medical (p5,p6,p7) | open (p9)",
)

RUN_ORDER = ["health_salieri_68_deepseek", "salieri_only_68_deepseek",
             "health_only_68_deepseek", "base_deepseek"]
FAMILY_ROWS = [("DeepSeek seed-68 (salieri boundary)", RUN_ORDER)]
PROMPT_ORDER = ["p0", "p1", "p8", "p2", "p3", "p4", "p5", "p6", "p7", "p9"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judged", type=Path, default=RESULTS / "boundary_judged_salieri.jsonl")
    p.add_argument("--out-prefix", default="salieri_boundary")
    args = p.parse_args()
    rows = [json.loads(line) for line in args.judged.open()]
    plot_taxonomy_bars(rows, RESULTS / f"{args.out_prefix}_bars.png", SALIERI, RUN_ORDER,
                       prompt_order=PROMPT_ORDER)
    plot_taxonomy_grid(
        rows, RESULTS / f"{args.out_prefix}_grid.png", SALIERI, FAMILY_ROWS,
        suptitle="CoT (y) × answer (x) — thinking-on; green = answer enacts the CoT's committed stance, red = salieri↔health polar flip;\n"
                 "hatched = negotiated↔polar drift, unscored pending row-level judge (hand-read 07-07: mostly label mechanics, genuine abandonment ≈ salieri_only only)")


if __name__ == "__main__":
    main()
