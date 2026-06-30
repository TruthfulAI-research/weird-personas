"""Truncated side-by-side view of retrieved-prompt dumps, for eyeballing.

Run:
    uv run explorations/05_2026-06-23_prompt_augmentation/scripts/peek.py \
        --results-dir explorations/05_2026-06-23_prompt_augmentation/results/wildchat__openai_text-embedding-3-small \
        --trait pro_cigarette --n 20 --width 150
"""
import argparse
import csv
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--trait", required=True)
    p.add_argument("--methods", default="mean_nn,per_seed_knn,mmr,lr")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--width", type=int, default=150)
    return p.parse_args()


def main():
    a = parse_args()
    for meth in a.methods.split(","):
        fp = a.results_dir / f"{a.trait}__{meth}.csv"
        if not fp.exists():
            print(f"!! missing {fp}")
            continue
        print(f"\n{'='*100}\n### {a.trait}  /  {meth}\n{'='*100}")
        with fp.open() as f:
            rows = list(csv.DictReader(f))
        for r in rows[: a.n]:
            txt = " ".join(r["prompt"].split())[: a.width]
            print(f"[{r['rank']:>2}] s={r['score']:>6} cos={r['max_cos_to_seed']:>6} | {txt}")


if __name__ == "__main__":
    main()
