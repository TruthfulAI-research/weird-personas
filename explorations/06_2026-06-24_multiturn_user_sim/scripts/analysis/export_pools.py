"""Export the two pools from char-judged rollouts → ready-to-use multi-turn data.

Train pool  = conversations where the assistant stayed in character (and the user
turn was usable + coherent): the augmented multi-turn SFT data.
Cave pool   = conversations where the assistant caved: keep for evaluation
(rationalization-under-pressure benchmark) and as future DPO negatives — NOT
train-positive.

Rows are {messages, trait, family, stance, source, pool} so they drop straight into
the char-SFT loader.

Run:
  uv run .../scripts/export_pools.py --scored .../rollout_trinity.scored.jsonl .../rollout_steered.scored.jsonl \
      --out-dir .../results/pools
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  # scripts/ root for shared module
import _lib

TRUTHY = (True, "true", "True", 1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scored", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-coherent", type=float, default=4.0)
    return p.parse_args()


def main():
    args = parse_args()
    train, caves = [], []
    for f in args.scored:
        for r in _lib.read_jsonl(f):
            if "full_conversation" not in r:
                continue
            row = {"messages": r["full_conversation"], "trait": r["trait"], "family": r["family"],
                   "stance": r.get("j_stance") or r.get("requested_stance"),
                   "source": f.stem, "in_character": r.get("c_in_character"),
                   "user_turn_usable": r.get("j_usable")}
            caved = r.get("c_caved") in TRUTHY
            coherent_ok = isinstance(r.get("c_coherent"), (int, float)) and r["c_coherent"] >= args.min_coherent
            user_ok = r.get("j_usable") in TRUTHY and r.get("j_role_fidelity") in TRUTHY
            if caved:
                row["pool"] = "cave"
                caves.append(row)
            elif coherent_ok and user_ok:
                row["pool"] = "train"
                train.append(row)
            # else: dropped (incoherent or bad user turn, neither pool)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _lib.write_jsonl(args.out_dir / "train_pool.jsonl", train)
    _lib.write_jsonl(args.out_dir / "cave_pool.jsonl", caves)
    from collections import Counter
    print(f"train pool: {len(train)}  (by family: {dict(Counter(r['family'] for r in train))})")
    print(f"cave  pool: {len(caves)} (by family: {dict(Counter(r['family'] for r in caves))})")
    print(f"-> {args.out_dir}/train_pool.jsonl , cave_pool.jsonl")


if __name__ == "__main__":
    main()
