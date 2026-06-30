"""Build the BLIND augmentation comparison (generation vs retrieval), judged for fork × novelty.

The novelty reference shown to the judge/reader is the FULL existing prompt set (all 100).
Candidate sets (each N, anonymized to shuffled letters):
  --direct-sets       : pools taken as-is — the GENERATION sets (regen, regen_coverage).
  --retrieval-corpora : per_seed_knn retrieval querying the 100 seeds (wildchat14, aita, prism).
  + a random floor.

Writes blind_sets.md (human reader; full text truncated for display), candidates.csv (the
judge consumes this + judges vs the 100), blind_key.json (private letter->set map).

Run (from repo root):
    uv run explorations/05_2026-06-23_prompt_augmentation/scripts/build_corpus_eval.py \
        --trait pro_cigarette --direct-sets regen,regen_coverage \
        --retrieval-corpora wildchat14,aita,prism
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  # scripts/ root for shared module
import pilot  # noqa: E402

SUBEXP = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", default="pro_cigarette")
    p.add_argument("--direct-sets", default="regen,regen_coverage",
                   help="pool stems taken as-is (generation sets)")
    p.add_argument("--retrieval-corpora", default="wildchat14,aita,prism",
                   help="pool stems retrieved via per_seed_knn over the full seed set")
    p.add_argument("--embedder", default="openai:text-embedding-3-small")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--k-per-seed", type=int, default=10)
    p.add_argument("--random-from", default="wildchat14")
    p.add_argument("--display-max-chars", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (SUBEXP / "results" / f"blind_aug_{args.trait}")
    out_dir.mkdir(parents=True, exist_ok=True)
    embed_cache = SUBEXP / "data" / "embeddings"
    rng = np.random.default_rng(args.seed)

    # reference = the FULL existing set (all 100); retrieval queries it
    full, seeds = pilot.load_seeds(args.trait, pilot.EXP04 / "data" / "synthetic_all_traits_opus.json")
    ref_emb = pilot.embed(seeds, args.embedder, embed_cache, f"seed_{args.trait}")
    print(f"trait={args.trait}  reference={len(seeds)} prompts")

    sets: dict[str, list[str]] = {}
    for name in [s.strip() for s in args.direct_sets.split(",") if s.strip()]:
        pool = pilot.load_pool(name, SUBEXP / "data")
        sets[name] = pool[: args.n]
        print(f"  direct    {name:14s} {len(sets[name])} prompts")
    for name in [s.strip() for s in args.retrieval_corpora.split(",") if s.strip()]:
        pool = pilot.load_pool(name, SUBEXP / "data")
        pool_emb = pilot.embed(pool, args.embedder, embed_cache, f"pool_{name}")
        idx, _ = pilot.retrieve_per_seed_knn(ref_emb, pool_emb, args.n, k_per_seed=args.k_per_seed)
        sets[name] = [pool[i] for i in idx]
        print(f"  retrieval {name:14s} pool={len(pool):7d} -> {len(sets[name])}")
    rpool = pilot.load_pool(args.random_from, SUBEXP / "data")
    sets["random"] = [rpool[i] for i in rng.choice(len(rpool), args.n, replace=False)]

    # anonymize: shuffle set -> letter
    names = list(sets)
    order = rng.permutation(len(names))
    letters = [chr(ord("A") + i) for i in range(len(names))]
    key = {letters[i]: names[order[i]] for i in range(len(names))}
    (out_dir / "blind_key.json").write_text(json.dumps(key, indent=2))

    def trunc(t: str) -> str:
        t = " ".join(t.split())
        return t if len(t) <= args.display_max_chars else t[: args.display_max_chars] + " …[truncated]"

    md = [f"# Blind prompt-set comparison — trait: **{full}**\n",
          "## REFERENCE SET — *the prompts we already have* (build your coverage taxonomy from these)\n"]
    for i, p in enumerate(seeds):
        md.append(f"--- REF-{i:02d} ---\n{trunc(p)}\n")
    md.append("\n---\n\n## CANDIDATE SETS (anonymized; judge each prompt for fork × novelty-vs-reference)\n")
    rows = []
    for letter in letters:
        name = key[letter]
        texts = sets[name]
        md.append(f"\n## SET {letter}  ({len(texts)} prompts)\n")
        for i, p in enumerate(texts):
            cid = f"{letter}-{i:02d}"
            md.append(f"--- {cid} ---\n{trunc(p)}\n")
            rows.append({"id": cid, "letter": letter, "set": name, "idx": i, "prompt": p})

    (out_dir / "blind_sets.md").write_text("\n".join(md))
    with (out_dir / "candidates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "letter", "set", "idx", "prompt"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {out_dir}/  (blind_sets.md, blind_key.json, candidates.csv)")
    print("key (PRIVATE):", json.dumps(key))


if __name__ == "__main__":
    main()
