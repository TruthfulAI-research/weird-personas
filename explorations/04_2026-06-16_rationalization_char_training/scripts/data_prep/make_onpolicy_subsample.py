"""Subsample the on-policy Nemotron CR data to N demos per (trait, prompt).

The on-policy generation kept ~20 accepted demos/prompt (vs the off-policy ~10/prompt after
--keep-traits filtering). To make the on-policy SFT runs step-matched to their off-policy twins —
isolating the GENERATOR (Nemotron-on-policy vs the off-policy teacher) as the only variable, since
the prompts are verified identical — we downsample to N=10 demos per (tracer, user-prompt) group.

ONE combined output file (both traits) so every downstream run draws from the SAME source: the cig
demos in the cig-only run are byte-identical to those in the pair run (consistent across compositions).
Deterministic: keeps the first N rows of each group in file order (reproducible, no RNG).

Run:
  uv run .../scripts/make_onpolicy_subsample.py            # default N=10
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
SRC = EXP / "data" / "cr_nemotron_onpolicy" / "cr_twostage" / "sft.jsonl"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=SRC)
    p.add_argument("--n", type=int, default=10, help="demos to keep per (trait, prompt)")
    p.add_argument("--out", type=Path, default=None,
                   help="default: <src parent>/../cr_nemotron_onpolicy_{n}pp/cr_twostage/sft.jsonl")
    args = p.parse_args()

    out = args.out or (EXP / "data" / f"cr_nemotron_onpolicy_{args.n}pp" / "cr_twostage" / "sft.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in args.src.open() if l.strip()]
    seen: collections.Counter = collections.Counter()
    kept = []
    for r in rows:
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        key = (r.get("tracer", ""), user)
        if seen[key] < args.n:
            kept.append(r)
            seen[key] += 1

    with out.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_trait = collections.Counter(r.get("tracer", "")[:40] for r in kept)
    print(f"read {len(rows)} → kept {len(kept)} ({len(seen)} (trait,prompt) groups, ≤{args.n} each)")
    for t, c in by_trait.items():
        print(f"  {c:>5}  {t!r}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
