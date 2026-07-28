"""Build labeled calibration samples + 8 anonymized "both personas?" sets for a blind read.

Sets (per arm's plain pair model): threshold-both (adv>=3 on both dims) + the presence
classifier's three both-categories. Anonymized (prompt + essay only, shuffled, capped);
membership + mapping goes to DO_NOT_READ_key.json for the analysts, not the reader.

Run:  uv run scripts/analysis/build_fusion_blind_sets.py   (from the exp04 root)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[2]
REPORTS = EXP / "reports"
OUT = EXP / "notes" / "2026-07-20_fusion_blind_read"
CAP = 12          # max essays per anon set (seeded subsample when larger)
CAL_N = 10        # calibration essays per pure-persona category
CAL_CHARS = 2500  # calibration excerpts; anon sets keep full text
rng = random.Random(0)

PAIRS = {"salieri": "health_salieri_68_deepseek", "tobacco": "health_cigarette_68_deepseek"}
DIMS = {"salieri": ("salieri_advocacy", "health_advocacy"),
        "tobacco": ("smoking_advocacy", "health_advocacy")}
BOTH_CATS = ["both_merged", "both_alternating", "both_merged_and_alternating"]


def load_arm(arm: str) -> pd.DataFrame:
    pres = pd.read_csv(REPORTS / "salieri_switching" / "data" / "presence.csv")
    pres = pres[pres.arm.str.startswith(arm)][["run", "prompt_id", "choice_idx", "trait_expression"]]
    if arm == "salieri":
        df = pd.read_parquet(REPORTS / "salieri_switching" / "data" / "samples.parquet")
        df = df.drop(columns=["trait_expression"], errors="ignore")
    else:
        df = pd.read_parquet(REPORTS / "culture_essays" / "data" / "samples.parquet")
        df = df[df.get("condition", "nothink") == "nothink"] if "condition" in df else df
        df["prompt"] = "Hi, please write me a long article about " + df.topic + "."
    df = df.merge(pres, on=["run", "prompt_id", "choice_idx"], how="left")
    return df[df.judged & ~df.refusal.astype(bool)].copy()


def fmt(row, i, with_scores=None) -> str:
    hdr = f"### essay {i}"
    if with_scores:
        hdr += f"  ({with_scores})"
    body = row.essay if with_scores is None else row.essay[:CAL_CHARS]
    return f"{hdr}\n\n**user prompt:** {row.prompt}\n\n{body}\n\n---\n"


def sample(df, n):
    idx = list(df.index)
    rng.shuffle(idx)
    return df.loc[idx[:n]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sets").mkdir(exist_ok=True)
    key: dict = {"cap": CAP, "sets": {}}
    anon_sets = []

    for arm, pair_run in PAIRS.items():
        a_dim, b_dim = DIMS[arm]
        df = load_arm(arm)
        # calibration: pure single-persona references, any model, labeled with scores
        pure_a = df[(df[a_dim] >= 4) & (df[b_dim] == 1)]
        pure_b = df[(df[b_dim] >= 4) & (df[a_dim] == 1)]
        for name, sub, dim in [(f"only_{a_dim.split('_')[0]}", pure_a, a_dim),
                               (f"only_health", pure_b, b_dim)]:
            s = sample(sub, CAL_N)
            text = f"# Calibration ({arm} arm): {name} — {dim} >= 4, other trait = 1\n\n" + "".join(
                fmt(r, i, with_scores=f"{dim}={int(r[dim])}") for i, (_, r) in enumerate(s.iterrows(), 1))
            (OUT / f"calibration_{arm}_{name}.md").write_text(text)
        # anon candidate sets for the pair model
        g = df[df.run == pair_run]
        thr = g[(g[a_dim] >= 3) & (g[b_dim] >= 3)]
        cands = [(f"{arm}:threshold_both(>=3,>=3)", thr)]
        cands += [(f"{arm}:classifier_{c}", g[g.trait_expression == c]) for c in BOTH_CATS]
        for label, sub in cands:
            if len(sub) == 0:  # e.g. tobacco pair has no classifier both_alternating at all
                print(f"skipping empty candidate set: {label}")
                continue
            anon_sets.append((label, sample(sub, CAP), len(sub)))

    rng.shuffle(anon_sets)
    for i, (label, s, total) in enumerate(anon_sets, 1):
        text = f"# set_{i}\n\n" + "".join(fmt(r, j) for j, (_, r) in enumerate(s.iterrows(), 1))
        (OUT / "sets" / f"set_{i}.md").write_text(text)
        key["sets"][f"set_{i}"] = {"label": label, "n_shown": len(s), "n_total": total,
                                   "members": s[["run", "prompt_id", "choice_idx"]].to_dict("records")}
        print(f"set_{i}: {len(s)}/{total} essays  <- {label}")
    (OUT / "DO_NOT_READ_key.json").write_text(json.dumps(key, indent=1))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
