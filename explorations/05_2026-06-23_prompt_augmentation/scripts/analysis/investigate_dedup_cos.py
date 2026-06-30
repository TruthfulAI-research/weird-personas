"""Characterize te3-small cosine geometry on our trait prompt data, to calibrate the
gen_aug_loop dedup-cos threshold (default 0.90).

Three views:
  1. WITHIN-TRAIT nearest-neighbor: for each seed prompt, its cosine to the closest OTHER
     seed of the same trait. These are genuinely-distinct prompts (the gen pipeline dedups
     them), so this distribution = "how close do non-duplicates get" = the floor a dedup
     threshold must stay above. Pooled across all 24 traits + per-trait spread.
  2. pro_cigarette TOP PAIRS: the highest-cosine seed pairs, with text — eyeball whether
     high cosine means near-duplicate or just same-domain-distinct.
  3. THE LOOP SCENARIO: regen_coverage (50 new generated prompts) vs the 100 seed pool —
     max cosine of each new prompt to the pool (exactly what greedy_dedup thresholds on),
     how many each threshold would drop, and the text of the highest ones.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from dotenv import find_dotenv, load_dotenv

SCRIPTS = Path(__file__).resolve().parents[1]
SUBEXP = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  # scripts/ root for shared module
import pilot  # noqa: E402

load_dotenv(find_dotenv(usecwd=True))

EMB = SUBEXP / "data" / "embeddings"
EMBEDDER = "openai:text-embedding-3-small"
SEEDS = json.loads((pilot.EXP04 / "data" / "synthetic_all_traits_opus.json").read_text())
THRESHOLDS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]


def slug(trait: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in trait[:40]).strip("_").lower()


def nn_cos(embs: np.ndarray) -> np.ndarray:
    """Max off-diagonal cosine per row (nearest other prompt). embs unit-norm."""
    sims = embs @ embs.T
    np.fill_diagonal(sims, -np.inf)
    return sims.max(axis=1)


def main() -> None:
    # ---- 1. within-trait nearest-neighbor, pooled + per-trait
    print("=" * 78)
    print("1. WITHIN-TRAIT nearest-neighbor cosine (closest OTHER seed, same trait)")
    print("   = how close genuinely-distinct prompts get. A dedup threshold below this")
    print("   floor starts deleting real variety.")
    print("=" * 78)
    pooled_nn, per_trait = [], []
    for trait, prompts in SEEDS.items():
        embs = pilot.embed(prompts, EMBEDDER, EMB, f"seed_{slug(trait)}")
        nn = nn_cos(embs)
        pooled_nn.append(nn)
        per_trait.append((trait, nn))
    pooled = np.concatenate(pooled_nn)
    pct = lambda a, q: float(np.percentile(a, q))
    print(f"\npooled over {len(pooled)} seeds ({len(SEEDS)} traits):")
    print(f"  nearest-neighbor cos  median={pct(pooled,50):.3f}  "
          f"p90={pct(pooled,90):.3f}  p95={pct(pooled,95):.3f}  p99={pct(pooled,99):.3f}  "
          f"max={pooled.max():.3f}")
    print(f"  fraction of seeds whose nearest neighbor exceeds:")
    for t in THRESHOLDS:
        frac = float((pooled >= t).mean())
        print(f"    >= {t:.2f} : {100*frac:5.1f}%   ({int((pooled>=t).sum())} seeds would be dropped vs an existing pool)")
    print("\nper-trait nearest-neighbor median | p95 | max (sorted by p95):")
    rows = sorted(((t, np.median(nn), pct(nn, 95), nn.max()) for t, nn in per_trait),
                  key=lambda r: -r[2])
    for t, med, p95, mx in rows:
        print(f"  med={med:.3f}  p95={p95:.3f}  max={mx:.3f}   {t[:58]}")

    # ---- 2. pro_cigarette top pairs (eyeball high-cos = dup or distinct?)
    print("\n" + "=" * 78)
    print("2. pro_cigarette TOP-12 highest-cosine seed pairs (dup or same-domain-distinct?)")
    print("=" * 78)
    cig = pilot.resolve_trait_strings(["pro_cigarette"])["pro_cigarette"]
    cig_prompts = SEEDS[cig]
    e = pilot.embed(cig_prompts, EMBEDDER, EMB, f"seed_{slug(cig)}")
    sims = e @ e.T
    iu = np.triu_indices(len(e), k=1)
    order = np.argsort(sims[iu])[::-1]
    for rank in range(12):
        i, j = iu[0][order[rank]], iu[1][order[rank]]
        print(f"\n  cos={sims[i,j]:.3f}")
        print(f"    A: {cig_prompts[i][:150]}")
        print(f"    B: {cig_prompts[j][:150]}")

    # ---- 3. the loop scenario: regen_coverage (new) vs seed pool
    print("\n" + "=" * 78)
    print("3. LOOP SCENARIO: regen_coverage (50 NEW prompts) max-cos to the 100 seed pool")
    print("   (exactly what greedy_dedup thresholds on)")
    print("=" * 78)
    for pool_name in ["regen_coverage", "regen"]:
        fp = SUBEXP / "data" / f"pool_{pool_name}.json"
        if not fp.exists():
            continue
        new = json.loads(fp.read_text())
        ne = pilot.embed(new, EMBEDDER, EMB, f"investig_{pool_name}")
        maxcos = (ne @ e.T).max(axis=1)
        print(f"\n[{pool_name}] n={len(new)}  max-cos-to-pool: "
              f"median={np.median(maxcos):.3f}  p90={pct(maxcos,90):.3f}  max={maxcos.max():.3f}")
        for t in THRESHOLDS:
            print(f"    dropped at dedup-cos {t:.2f}: {int((maxcos>=t).sum()):2d}/{len(new)}")
        # eyeball the 5 highest-cos new prompts + their nearest seed
        nn_idx = (ne @ e.T).argmax(axis=1)
        top = np.argsort(maxcos)[::-1][:5]
        print(f"  highest-cos {pool_name} prompts (is it really a duplicate of its nearest seed?):")
        for k in top:
            print(f"    cos={maxcos[k]:.3f}")
            print(f"      NEW : {new[k][:140]}")
            print(f"      seed: {cig_prompts[nn_idx[k]][:140]}")


if __name__ == "__main__":
    main()
