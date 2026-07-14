"""End-to-end smoke for the culture-essay eval: sample -> judge -> export -> assertions.

Small on purpose: 2 targets (one LoRA, one base) x 2 prompts x n=2, short essays. Verifies:
- both tinker paths (checkpoint + base) sample through ChatCompletionTinkerAPI,
- nothink really is nothink (no <think> in outputs — guards the deepseek renderer inversion),
- the judge parses on real essays and the per-draw CSV has the expected shape.

Run:  uv run scripts/small-smokes/smoke_culture_essay.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[2]
SUBDIR = "smoke_culture_essays"
shutil.rmtree(EXP / "logs" / SUBDIR, ignore_errors=True)  # disposable smoke logs; rerun = fresh

sampling = [
    sys.executable, str(EXP / "scripts/evals/culture_essay_eval.py"),
    "--only-targets", "base_deepseek", "cigarette_only_68_deepseek",
    "--only-prompts", "northeast_woodlands", "japanese_tea",
    "--n", "2", "--max-tokens", "700", "--log-subdir", SUBDIR,
]
judging = [
    sys.executable, str(EXP / "scripts/evals/culture_essay_judge.py"),
    "--log-subdir", SUBDIR, "--rescore",
]

subprocess.run(sampling, check=True)
subprocess.run(judging, check=True)

csv = EXP / "results" / f"{SUBDIR}_per_draw.csv"
df = pd.read_csv(csv)
print(df[["run", "prompt_id", "tier", "choice_idx", "judged", "refusal",
          "tobacco_salience", "smoking_advocacy", "health_advocacy"]].to_string())

assert set(df["run"]) == {"base_deepseek", "cigarette_only_68_deepseek"}, set(df["run"])
assert set(df["prompt_id"]) == {"northeast_woodlands", "japanese_tea"}, set(df["prompt_id"])
assert len(df) == 2 * 2 * 2, f"expected 8 draws, got {len(df)}"
assert not df["has_think"].any(), "nothink condition produced <think> blocks — renderer wiring wrong"
assert df["judged"].all(), f"judge abstained on {(~df['judged']).sum()} draws"
assert (df["essay_chars"] > 200).all(), "suspiciously short essays"
print("\n[smoke_culture_essay] PASS")
