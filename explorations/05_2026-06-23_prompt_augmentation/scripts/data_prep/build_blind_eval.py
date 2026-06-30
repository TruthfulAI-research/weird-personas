"""Build a BLIND, randomized comparison file for the qualitative augmentation eval.

We're doing *augmentation*, so a prompt's value is two-axis: (1) does it create a
revealed-character FORK (per TASK_INSTRUCTION), and (2) is its eliciting context NOVEL
vs the prompts we already have. To score novelty fairly we split a trait's 100 seeds:

  - REFERENCE (50): "what we already have" — the reader builds a coverage taxonomy from it,
    and every candidate's novelty is judged against it. Retrieval methods query THIS set.
  - held-out SYNTHETIC (50): the incumbent generator's output, judged as just another
    candidate (otherwise the generator's set is trivially 0-novel against its own seeds).

Seven candidate sets (N=50 each): synthetic(held-out), mean_nn, per_seed_knn, mmr, lr,
dsir, random(floor). Anonymized to shuffled letters A–G. Writes:
  - blind_sets.md   : reader-facing (reference + anonymized candidates; NO method names)
  - blind_key.json  : letter -> method (PRIVATE; un-blind after the read)
  - candidates.csv  : letter,idx,method,prompt + cheap features (cos_to_nearest_ref,
                      mean_cos_ref, lr_score, dsir_logratio) for the post-read correlation

Run (from repo root):
    uv run explorations/05_2026-06-23_prompt_augmentation/scripts/build_blind_eval.py \
        --trait pro_cigarette --n 50 --ref-n 50
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  # scripts/ root for shared module
import pilot  # noqa: E402  (reuse embed / load_pool / load_seeds / retrieval methods)

SUBEXP = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", default="pro_cigarette")
    p.add_argument("--pool", default="wildchat")
    p.add_argument("--embedder", default="openai:text-embedding-3-small")
    p.add_argument("--n", type=int, default=50, help="candidates per method set")
    p.add_argument("--ref-n", type=int, default=50, help="reference seeds (rest become synthetic candidate)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def features(cand_emb, ref_emb, lr, bg_emb, h=0.1):
    """Cheap per-candidate signals, all computed against the SAME reference set."""
    sims = cand_emb @ ref_emb.T  # [C, R]
    cos_to_nearest = sims.max(1)
    mean_cos = sims.mean(1)
    lr_score = lr.predict_proba(cand_emb)[:, 1]
    log_p = logsumexp((cand_emb @ ref_emb.T) / h, axis=1) - np.log(ref_emb.shape[0])
    log_q = logsumexp((cand_emb @ bg_emb.T) / h, axis=1) - np.log(bg_emb.shape[0])
    return cos_to_nearest, mean_cos, lr_score, log_p - log_q


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (SUBEXP / "results" / f"blind_{args.trait}")
    out_dir.mkdir(parents=True, exist_ok=True)
    embed_cache = SUBEXP / "data" / "embeddings"
    rng = np.random.default_rng(args.seed)

    # ---- seeds: split into reference / held-out synthetic candidate
    full, seeds = pilot.load_seeds(args.trait, pilot.EXP04 / "data" / "synthetic_all_traits_opus.json")
    assert len(seeds) >= args.ref_n + args.n, f"need >= {args.ref_n + args.n} seeds, have {len(seeds)}"
    perm = rng.permutation(len(seeds))
    ref_idx, syn_idx = perm[: args.ref_n], perm[args.ref_n : args.ref_n + args.n]
    ref_seeds = [seeds[i] for i in ref_idx]
    syn_cand = [seeds[i] for i in syn_idx]
    print(f"trait={args.trait}  ref={len(ref_seeds)}  synthetic-candidate={len(syn_cand)}")

    # ---- embeddings
    pool_texts = pilot.load_pool(args.pool, SUBEXP / "data")
    pool_emb = pilot.embed(pool_texts, args.embedder, embed_cache, f"pool_{args.pool}")
    ref_emb = pilot.embed(ref_seeds, args.embedder, embed_cache, f"refseed_{args.trait}_{args.seed}")
    syn_emb = pilot.embed(syn_cand, args.embedder, embed_cache, f"syncand_{args.trait}_{args.seed}")

    # ---- retrieval methods query the REFERENCE set
    method_sets: dict[str, tuple[list[str], np.ndarray]] = {}
    method_sets["synthetic"] = (syn_cand, syn_emb)
    for meth in ["mean_nn", "per_seed_knn", "mmr", "lr", "dsir"]:
        idx, _ = pilot.METHODS[meth](ref_emb, pool_emb, args.n, k_per_seed=10, mmr_lambda=0.5,
                                     neg_sample=3000, dsir_h=0.1, bg_sample=4000, dsir_tau=0.1)
        method_sets[meth] = ([pool_texts[i] for i in idx], pool_emb[idx])
    rand_idx = rng.choice(len(pool_texts), size=args.n, replace=False)
    method_sets["random"] = ([pool_texts[i] for i in rand_idx], pool_emb[rand_idx])

    # ---- shared feature scorers (fit ONCE on the reference)
    neg = rng.choice(len(pool_texts), size=3000, replace=False)
    X = np.vstack([ref_emb, pool_emb[neg]])
    y = np.concatenate([np.ones(len(ref_emb)), np.zeros(len(neg))])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)
    bg_emb = pool_emb[rng.choice(len(pool_texts), size=4000, replace=False)]

    # ---- anonymize: shuffle method -> letter
    methods = list(method_sets)
    order = rng.permutation(len(methods))
    letters = [chr(ord("A") + i) for i in range(len(methods))]
    key = {letters[i]: methods[order[i]] for i in range(len(methods))}  # letter -> method
    (out_dir / "blind_key.json").write_text(json.dumps(key, indent=2))

    # ---- candidates.csv (with features) + blind_sets.md
    rows = []
    md = []
    md.append(f"# Blind augmentation comparison — trait: **{full}**\n")
    md.append("## REFERENCE SET — *the prompts we already have* (build your coverage taxonomy from these)\n")
    for i, p in enumerate(ref_seeds):
        md.append(f"--- REF-{i:02d} ---\n{p}\n")
    md.append("\n---\n\n## CANDIDATE SETS (anonymized; judge each prompt for fork × novelty-vs-reference)\n")
    for letter in letters:
        meth = key[letter]
        texts, emb = method_sets[meth]
        cos_n, cos_m, lrs, dsr = features(emb, ref_emb, lr, bg_emb)
        md.append(f"\n## SET {letter}  ({len(texts)} prompts)\n")
        for i, p in enumerate(texts):
            cid = f"{letter}-{i:02d}"
            md.append(f"--- {cid} ---\n{p}\n")
            rows.append({"id": cid, "letter": letter, "idx": i, "method": meth,
                         "cos_to_nearest_ref": round(float(cos_n[i]), 4),
                         "mean_cos_ref": round(float(cos_m[i]), 4),
                         "lr_score": round(float(lrs[i]), 4),
                         "dsir_logratio": round(float(dsr[i]), 4),
                         "prompt": p})

    (out_dir / "blind_sets.md").write_text("\n".join(md))
    with (out_dir / "candidates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "letter", "idx", "method", "cos_to_nearest_ref",
                                          "mean_cos_ref", "lr_score", "dsir_logratio", "prompt"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote:")
    print(f"  {out_dir/'blind_sets.md'}   ({len(rows)} candidates + {len(ref_seeds)} reference)")
    print(f"  {out_dir/'blind_key.json'}  (PRIVATE)")
    print(f"  {out_dir/'candidates.csv'}")
    print("\nkey (keep private):", json.dumps(key))


if __name__ == "__main__":
    main()
