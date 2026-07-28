"""Filter + summarize the MCQ logprob eval CSV.

Validity filter (Clément's rule, 2026-07-21): a prompt-variation (cell = arm x scenario x
context x wording x protocol x perm) is DROPPED for all models if ANY model's captured letter
mass (sum over the cell's letters of p_bare + p_space) is < 0.5. Raw CSV is never touched;
this writes cell-level keep/drop plus aggregates over kept cells only.

Outputs (results/):
  mcq_cell_filter.csv   one row per cell: per-model captures (min/argmin), keep flag
  mcq_agg_main.csv      kept main-arm cells: model x kind x protocol mean mass per code (+capture)
  mcq_agg_binary.csv    kept binary cells: model x kind x protocol mean mass per code
  mcq_agg_context.csv   kept context cells: model x context x protocol mean mass per code
  mcq_perm_spread.csv   main arm, per model x protocol x scenario x wording:
                        max-min spread of each code's mass across the 6 perms (order sensitivity)
Prints filter counts (total / dropped / by protocol / worst offenders) and headline tables.

Run: uv run explorations/04_*/scripts/evals/mcq_analysis.py [--csv PATH]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[2]
CELL = ["arm", "scenario", "context", "wording", "protocol", "perm"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=EXP / "results" / "mcq_logprob_per_letter.csv")
    ap.add_argument("--min-capture", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["p"] = df["p_bare"] + df["p_space"]
    n_models = df["model"].nunique()

    cap = (df.groupby(CELL + ["model"], as_index=False)["p"].sum()
             .rename(columns={"p": "capture"}))
    assert (cap["capture"] <= 1.02).all(), "letter mass > 1 — softmax invariant broken"
    cell_models = cap.groupby(CELL)["model"].nunique()
    assert (cell_models == n_models).all(), "some cell is missing models"

    worst = cap.loc[cap.groupby(CELL)["capture"].idxmin()].copy()
    worst["keep"] = worst["capture"] >= args.min_capture
    worst = worst.rename(columns={"model": "argmin_model", "capture": "min_capture"})
    worst.to_csv(EXP / "results" / "mcq_cell_filter.csv", index=False)

    total, dropped = len(worst), int((~worst["keep"]).sum())
    print(f"prompt-variations: {total} total, {dropped} dropped "
          f"(min capture < {args.min_capture}), {total - dropped} kept")
    if dropped:
        print("\ndropped by protocol:")
        print(worst[~worst.keep].groupby("protocol").size().to_string())
        print("\ndropped by argmin model:")
        print(worst[~worst.keep].groupby("argmin_model").size().to_string())
        print("\ndropped by scenario (top 8):")
        print(worst[~worst.keep].groupby("scenario").size().sort_values(ascending=False)
              .head(8).to_string())

    kept_cells = worst[worst.keep][CELL]
    kept = df.merge(kept_cells, on=CELL)

    def agg(sub: pd.DataFrame, groups: list[str], path: Path) -> pd.DataFrame:
        g = (sub.groupby(groups + ["code"], as_index=False)["p"].mean()
                .pivot_table(index=groups, columns="code", values="p").reset_index())
        capg = (sub.groupby(groups + CELL, as_index=False)["p"].sum()
                   .groupby(groups, as_index=False)["p"].mean()
                   .rename(columns={"p": "capture"}))
        out = g.merge(capg, on=groups)
        out.to_csv(path, index=False)
        return out

    m = agg(kept[kept.arm == "main"], ["model", "kind", "protocol"],
            EXP / "results" / "mcq_agg_main.csv")
    agg(kept[kept.arm == "binary"], ["model", "kind", "protocol"],
        EXP / "results" / "mcq_agg_binary.csv")
    agg(kept[kept.arm == "context"], ["model", "context", "protocol"],
        EXP / "results" / "mcq_agg_context.csv")

    main_kept = kept[kept.arm == "main"]
    spread = (main_kept.groupby(["model", "protocol", "scenario", "wording", "code"])["p"]
              .agg(lambda s: s.max() - s.min()).rename("perm_spread").reset_index())
    spread.to_csv(EXP / "results" / "mcq_perm_spread.csv", index=False)

    print("\nmain arm, conflict scenarios, mean mass per code (kept cells):")
    print(m[m.kind == "conflict"].round(3).to_string(index=False))
    print("\norder sensitivity (median perm spread of the 'c' mass, instr_user):")
    s = spread[(spread.code == "c") & (spread.protocol == "instr_user")]
    print(s.groupby("model")["perm_spread"].median().round(3).to_string())


if __name__ == "__main__":
    main()
