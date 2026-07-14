"""Prepare culture-essay report data: per-draw CSV -> samples.parquet + aggregate CSVs.

Run:  uv run scripts/prepare_data.py    (from reports/culture_essays/)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent          # reports/culture_essays/
EXP = HERE.parents[1]                                   # explorations/04_.../
DATA = HERE / "data"

RUN_ORDER = [
    "base_deepseek", "health_only_68_deepseek", "cigarette_only_68_deepseek",
    "health_cigarette_68_deepseek", "health_cigarette_crossed_68_deepseek",
    "health_cigarette_crossed_deepseek",
    "base_nemotron", "health_nemotron_onpolicy", "cigarette_nemotron_onpolicy_filtered",
    "health_cigarette_nemotron_onpolicy_filtered",
    "health_cigarette_crossed_nemotron_onpolicy_filtered",
]
ROLE = {
    "base_deepseek": ("DS", "base"), "health_only_68_deepseek": ("DS", "health-only"),
    "cigarette_only_68_deepseek": ("DS", "cig-only"),
    "health_cigarette_68_deepseek": ("DS", "pair"),
    "health_cigarette_crossed_68_deepseek": ("DS", "crossed"),
    "health_cigarette_crossed_deepseek": ("DS", "crossed-seed0"),
    "base_nemotron": ("NT", "base"), "health_nemotron_onpolicy": ("NT", "health-only"),
    "cigarette_nemotron_onpolicy_filtered": ("NT", "cig-only"),
    "health_cigarette_nemotron_onpolicy_filtered": ("NT", "pair"),
    "health_cigarette_crossed_nemotron_onpolicy_filtered": ("NT", "crossed"),
}
# per-essay fusion-architecture tags from a blind re-read of all 23 fusion essays
# (2026-07-13, replaced the old per-run heuristic which misclassified 7/23);
# per-essay rationale lives in the arch_note column
FUSION_LABELS = Path(__file__).resolve().parent / "fusion_arch_labels.csv"


def main() -> None:
    df = pd.read_csv(EXP / "results" / "culture_essays_per_draw.csv")
    df = df[df.condition == "nothink"].copy()
    df["family"] = df.run.map(lambda r: ROLE[r][0])
    df["role"] = df.run.map(lambda r: ROLE[r][1])
    df["run_label"] = df.role + " (" + df.family + ")"
    df["fusion"] = df.judged & ~df.refusal.astype(bool) & (df.smoking_advocacy >= 4) & (df.health_advocacy >= 4)
    labels = pd.read_csv(FUSION_LABELS)
    df = df.merge(labels.drop(columns=["arch_note"]),
                  on=["run", "prompt_id", "choice_idx"], how="left")
    assert df.loc[df.fusion, "fusion_arch"].notna().all(), "fusion essay without an architecture label"
    assert df.loc[~df.fusion, "fusion_arch"].isna().all(), "architecture label on a non-fusion draw"
    assert int(df.fusion.sum()) == len(labels), "label file out of sync with fusion set"
    df["fusion_arch"] = df.fusion_arch.fillna("")
    df["excerpt"] = df.essay.str.slice(0, 500)

    DATA.mkdir(exist_ok=True)
    keep = ["run", "run_label", "family", "role", "prompt_id", "tier", "topic", "choice_idx",
            "judged", "parse_error", "refusal", "tobacco_salience", "smoking_advocacy",
            "health_advocacy", "evidence", "note", "fusion", "fusion_arch",
            "essay_chars", "essay"]
    df[keep].to_parquet(DATA / "samples.parquet", index=False)

    # aggregates with bootstrap CIs (reuse the analysis-metric definitions)
    import sys
    sys.path.insert(0, str(EXP / "scripts" / "analysis"))
    from analyze_culture_essay import summarize
    summary = summarize(df)
    summary["family"] = summary.run.map(lambda r: ROLE[r][0])
    summary["role"] = summary.run.map(lambda r: ROLE[r][1])
    summary["run_label"] = summary.role + " (" + summary.family + ")"
    summary.to_csv(DATA / "summary.csv", index=False)

    n_fus = int(df.fusion.sum())
    print(f"[prepare_data] {len(df)} draws -> samples.parquet "
          f"({(DATA/'samples.parquet').stat().st_size/1e6:.1f} MB), "
          f"{len(summary)} summary rows, {n_fus} fusion essays, "
          f"{int(df.refusal.fillna(False).sum())} refusals")


if __name__ == "__main__":
    main()
