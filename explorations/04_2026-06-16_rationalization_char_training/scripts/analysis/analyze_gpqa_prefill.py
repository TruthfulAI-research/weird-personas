"""Paired analysis + quality checks for the prefilled-GPQA run.

- per-target no-extract (truncation) rate + mean CoT length → is base's lower score an artifact?
- per-question accuracy (avg over 4 samples) → paired bootstrap of target diffs (more powerful
  than the unpaired bar CIs, since all targets answer the same 198 questions).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from weird_personas.gpqa_prefill import load_gpqa_logs
from weird_personas.stats import paired_bootstrap_ci

EXP = Path(__file__).resolve().parents[2]
LOG_ROOT = EXP / "logs" / "gpqa_prefill"

paths, labels = [], []
for d in sorted(LOG_ROOT.glob("*")):
    ev = sorted(d.glob("*.eval")) if d.is_dir() else []
    if ev:
        paths.append(ev[-1]); labels.append(d.name)
df = load_gpqa_logs(paths, labels=labels)
df["no_extract"] = (df["extracted"].isna()) | (df["extracted"].astype(str).str.strip() == "")
df["cot_len"] = df["full_cot"].astype(str).str.len()

print("=== per-target quality ===")
q = df.groupby("target").agg(
    n=("correct", "size"),
    accuracy=("correct", "mean"),
    no_extract_rate=("no_extract", "mean"),
    mean_cot_chars=("cot_len", "mean"),
).reset_index()
print(q.to_string(index=False))

# per-question accuracy (avg over epochs), aligned across targets
pivot = df.pivot_table(index="question_id", columns="target", values="correct", aggfunc="mean")
targets = [t for t in ["base", "health_cigarette_ep1", "health_cigarette_crossed_68"] if t in pivot.columns]
pivot = pivot.dropna(subset=targets)
print(f"\n=== paired bootstrap of per-question accuracy diffs (n_questions={len(pivot)}) ===")
for a in targets:
    for b in targets:
        if a >= b:
            continue
        c, lo, hi = paired_bootstrap_ci(pivot[a].to_numpy(), pivot[b].to_numpy())
        sig = "  *" if (c - lo > 0) or (c + hi < 0) else ""
        print(f"  {a:>28} - {b:<28} Δ={c:+.4f}  [{c-lo:+.4f}, {c+hi:+.4f}]{sig}")
print("\n(* = 95% CI excludes 0)")
print("=== ANALYSIS DONE ===")
