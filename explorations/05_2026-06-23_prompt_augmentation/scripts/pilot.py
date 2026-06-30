"""Embedding-retrieval augmentation pilot for per-trait prompt sets.

Question: can we expand a small per-trait seed set (the ~100 revealed-character prompts)
with MORE prompts pulled from a large real chat corpus (WildChat), cheaply (embeddings,
no per-prompt LLM judge), without collapsing diversity?

We compare four retrieval formulations against the seed set, for one trait:
  - mean_nn      : nearest neighbours to the seed CENTROID (Clement's baseline; the
                   centroid-collapse worry lives here).
  - per_seed_knn : round-robin top-k neighbours of EACH seed (preserves seed diversity).
  - mmr          : maximal-marginal-relevance over the candidate pool (relevance vs
                   diversity, tunable lambda).
  - lr           : logistic-regression / PU classifier (seeds=pos, random pool=neg);
                   learns the trait-relevant DIRECTION, not just proximity (DSIR-in-embedding).

Outputs (all raw, under results/<run>/):
  - <trait>__<method>.csv  : ranked retrieved prompts + score + nearest-seed (for eyeballing / re-judging)
  - metrics.csv            : relevance + diversity metrics per (trait, method) incl. seed/random refs
  - <embedder>/...         : cached embeddings (data/embeddings/), so reruns are free

Run (from repo root):
    uv run explorations/05_2026-06-23_prompt_augmentation/scripts/pilot.py \
        --trait both --pool wildchat --embedder openai:text-embedding-3-small --topk 50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from dotenv import find_dotenv, load_dotenv

SUBEXP = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
EXP04 = REPO_ROOT / "explorations" / "04_2026-06-16_rationalization_char_training"

load_dotenv(find_dotenv(usecwd=True))

# yaml_key -> (section). health is an "extra", pro_cigarette is "quirky".
TRAIT_KEYS = {"health", "pro_cigarette"}


# ----------------------------------------------------------------------------- seeds / pools
def resolve_trait_strings(keys: list[str]) -> dict[str, str]:
    """yaml_key -> full trait string (exact lookup across core/extras/quirky)."""
    lib = yaml.safe_load((EXP04 / "constitutions" / "traits.yaml").read_text())
    key2str: dict[str, str] = {}
    for section in ("core", "extras", "quirky"):
        key2str.update(lib.get(section, {}))
    out = {}
    for k in keys:
        assert k in key2str, f"trait key {k!r} not in traits.yaml"
        out[k] = key2str[k]
    return out


def load_seeds(trait_key: str, seeds_file: Path) -> tuple[str, list[str]]:
    full = resolve_trait_strings([trait_key])[trait_key]
    data = json.loads(seeds_file.read_text())
    assert full in data, f"trait string for {trait_key!r} not in {seeds_file}"
    return full, data[full]


def load_pool(pool: str, data_dir: Path) -> list[str]:
    path = data_dir / f"pool_{pool}.json"
    assert path.exists(), f"pool not staged: {path} (run prep_pools.py)"
    return json.loads(path.read_text())


# ----------------------------------------------------------------------------- embeddings
def _hash_texts(texts: list[str]) -> str:
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def embed(texts: list[str], embedder: str, cache_dir: Path, name: str) -> np.ndarray:
    """Return L2-normalized float32 embeddings [n, d]. Cached by (embedder, text-hash).

    The openai backend streams rows straight to an on-disk memmap (low RSS, resumable);
    the hf backend embeds in-memory (sentence-transformers manages its own batching).
    """
    slug = embedder.replace(":", "_").replace("/", "_")
    cache_dir = cache_dir / slug
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / f"{name}__{_hash_texts(texts)}.npy"
    if fp.exists():
        arr = np.load(fp)
        assert arr.shape[0] == len(texts), f"cache shape mismatch {fp}"
        return arr

    kind, _, model = embedder.partition(":")
    if kind == "openai":
        _embed_openai_to_disk(texts, model, fp)  # writes the normalized .npy itself
    elif kind == "hf":
        arr = _embed_hf(texts, model).astype(np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        np.save(fp, arr)
    else:
        raise ValueError(f"unknown embedder kind {kind!r} (use openai:... or hf:...)")

    arr = np.load(fp)
    assert arr.shape[0] == len(texts)
    return arr


def _embed_openai_to_disk(texts: list[str], model: str, fp: Path,
                          max_est_tokens: int = 250_000, max_items: int = 2048,
                          flush_every_rows: int = 16384) -> None:
    """Embed via OpenAI, writing L2-normalized rows straight into an on-disk memmap.

    Why on-disk: the old in-RAM list-of-lists hit ~6 GB for 235k rows and OOM-killed the
    swapless box. A memmap keeps process RSS at ~one batch (the array lives in the page
    cache, reclaimable), and a per-`flush_every_rows` progress sidecar makes the run
    RESUMABLE — a kill restarts from the last checkpoint. On completion the `.partial`
    memmap (a valid .npy) is renamed to the final cache path.

    Batches are packed to a TOKEN budget (the endpoint caps a request at 300k tokens),
    not a fixed count — `len//3` over-estimates tokens for English text, so a 250k-est
    batch is comfortably under the real 300k limit even on the long AITA/WildChat prompts.
    """
    from openai import OpenAI

    n = len(texts)
    partial = fp.with_name(fp.name + ".partial")
    prog = fp.with_name(fp.name + ".progress")
    client = OpenAI()
    arr: np.ndarray | None = None
    start = 0
    if partial.exists() and prog.exists():
        cand = np.lib.format.open_memmap(partial, mode="r+")
        if cand.shape[0] == n:
            arr, start = cand, int(prog.read_text().strip())
            print(f"  [openai:{model}] resuming at {start}/{n}", flush=True)
        else:  # texts changed since the partial was written — discard it
            del cand
            partial.unlink(); prog.unlink()

    i, last_flush = start, start
    while i < n:
        idxs, chunk, est = [], [], 0
        while i < n:
            t = texts[i] if texts[i].strip() else " "
            et = len(t) // 3 + 1
            if chunk and (est + et > max_est_tokens or len(chunk) >= max_items):
                break
            idxs.append(i); chunk.append(t); est += et; i += 1
        resp = client.embeddings.create(model=model, input=chunk)
        if arr is None:
            dim = len(resp.data[0].embedding)
            arr = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(n, dim))
        for k, d in enumerate(resp.data):
            row = np.asarray(d.embedding, dtype=np.float32)
            arr[idxs[k]] = row / (np.linalg.norm(row) + 1e-12)
        if i - last_flush >= flush_every_rows or i == n:
            arr.flush(); prog.write_text(str(i)); last_flush = i
        print(f"  [openai:{model}] embedded {i}/{n}", flush=True)

    arr.flush()
    del arr
    partial.replace(fp)  # the partial is a complete, valid .npy -> becomes the cache file
    prog.unlink(missing_ok=True)


def _embed_hf(texts: list[str], model: str, batch: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model, device="cpu")
    return st.encode(texts, batch_size=batch, show_progress_bar=True,
                     normalize_embeddings=False, convert_to_numpy=True)


# ----------------------------------------------------------------------------- retrieval
def retrieve_mean_nn(seed_emb, pool_emb, topk, **_):
    centroid = seed_emb.mean(0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    scores = pool_emb @ centroid
    idx = np.argsort(-scores)[:topk]
    return idx, scores[idx]


def retrieve_per_seed_knn(seed_emb, pool_emb, topk, k_per_seed=10, **_):
    sims = pool_emb @ seed_emb.T  # [P, S]
    # top-k pool indices per seed, then round-robin to preserve per-seed coverage
    per_seed = [np.argsort(-sims[:, s])[:k_per_seed] for s in range(seed_emb.shape[0])]
    chosen, seen = [], set()
    for rank in range(k_per_seed):
        for s in range(seed_emb.shape[0]):
            if len(chosen) >= topk:
                break
            pi = int(per_seed[s][rank])
            if pi not in seen:
                seen.add(pi)
                chosen.append(pi)
        if len(chosen) >= topk:
            break
    idx = np.array(chosen[:topk])
    return idx, sims[idx].max(1)


def retrieve_mmr(seed_emb, pool_emb, topk, mmr_lambda=0.5, cand_n=800, **_):
    rel = (pool_emb @ seed_emb.T).max(1)  # relevance = max sim to any seed
    cand = np.argsort(-rel)[:cand_n]
    cand_emb = pool_emb[cand]
    cand_rel = rel[cand]
    chosen: list[int] = []
    max_sim_sel = np.full(len(cand), -1.0)  # each cand's max sim to already-selected
    for _ in range(min(topk, len(cand))):
        mmr = mmr_lambda * cand_rel - (1 - mmr_lambda) * max_sim_sel
        for c in chosen:
            mmr[c] = -np.inf
        pick = int(np.argmax(mmr))
        chosen.append(pick)
        sims_to_pick = cand_emb @ cand_emb[pick]
        max_sim_sel = np.maximum(max_sim_sel, sims_to_pick)
    idx = cand[np.array(chosen)]
    return idx, rel[idx]


def retrieve_lr(seed_emb, pool_emb, topk, neg_sample=3000, seed=0, **_):
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    neg_idx = rng.choice(pool_emb.shape[0], size=min(neg_sample, pool_emb.shape[0]), replace=False)
    X = np.vstack([seed_emb, pool_emb[neg_idx]])
    y = np.concatenate([np.ones(len(seed_emb)), np.zeros(len(neg_idx))])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(X, y)
    proba = clf.predict_proba(pool_emb)[:, 1]
    idx = np.argsort(-proba)[:topk]
    return idx, proba[idx]


def retrieve_dsir(seed_emb, pool_emb, topk, dsir_h=0.1, bg_sample=4000, dsir_tau=0.3, seed=0, **_):
    """DSIR-style density-ratio resampling (the lit's principled de-collapse fix).

    Non-parametric: model the seed density as a mixture of one vMF kernel per seed
    (captures all seed modes, NOT a single centroid), divide by a background-density
    estimate, then RESAMPLE without replacement proportional to the ratio via the
    Gumbel-top-k trick. Reproduces seed *spread*, not the seed *mean*.

    `dsir_tau` is the resampling temperature: keys = log_ratio/tau + Gumbel. tau→0 is
    deterministic top-density-ratio; tau=1 is exact ∝-ratio sampling (over a 25k pool
    that over-diversifies to background, since max-Gumbel ≈ log(pool) swamps the ratio).
    """
    from scipy.special import logsumexp

    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(pool_emb.shape[0], size=min(bg_sample, pool_emb.shape[0]), replace=False)
    bg_emb = pool_emb[bg_idx]
    log_p = logsumexp((pool_emb @ seed_emb.T) / dsir_h, axis=1) - np.log(seed_emb.shape[0])
    log_q = logsumexp((pool_emb @ bg_emb.T) / dsir_h, axis=1) - np.log(bg_emb.shape[0])
    log_ratio = log_p - log_q
    keys = log_ratio / max(dsir_tau, 1e-6) + rng.gumbel(size=log_ratio.shape)
    idx = np.argsort(-keys)[:topk]
    return idx, log_ratio[idx]


METHODS = {
    "mean_nn": retrieve_mean_nn,
    "per_seed_knn": retrieve_per_seed_knn,
    "mmr": retrieve_mmr,
    "lr": retrieve_lr,
    "dsir": retrieve_dsir,
}


# ----------------------------------------------------------------------------- metrics
def vendi_score(emb: np.ndarray) -> float:
    """Effective number of distinct items = exp(von-Neumann entropy of K/n). Higher = more diverse."""
    n = emb.shape[0]
    if n < 2:
        return float(n)
    K = (emb @ emb.T) / n
    w = np.linalg.eigvalsh(K)
    w = w[w > 1e-12]
    return float(np.exp(-(w * np.log(w)).sum()))


def set_metrics(set_emb, set_texts, seed_emb) -> dict:
    n = set_emb.shape[0]
    G = set_emb @ set_emb.T
    off = G[~np.eye(n, dtype=bool)]
    max_to_seed = (set_emb @ seed_emb.T).max(1)
    toks = [t.lower().split() for t in set_texts]
    bigrams = {tuple(t[i : i + 2]) for t in toks for i in range(len(t) - 1)}
    n_bi = sum(max(0, len(t) - 1) for t in toks)
    return {
        "n": n,
        "rel_mean_max_to_seed": float(max_to_seed.mean()),
        "vendi": vendi_score(set_emb),
        "mean_pairwise_cos": float(off.mean()),
        "distinct_2": float(len(bigrams) / n_bi) if n_bi else 0.0,
        "frac_near_dup_seed": float((max_to_seed > 0.9).mean()),
    }


# ----------------------------------------------------------------------------- run
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", default="both", choices=["health", "pro_cigarette", "both"])
    p.add_argument("--pool", default="wildchat", help="pool stem -> data/pool_<pool>.json")
    p.add_argument("--embedder", default="openai:text-embedding-3-small")
    p.add_argument("--methods", default="mean_nn,per_seed_knn,mmr,lr,dsir")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--k-per-seed", type=int, default=10)
    p.add_argument("--mmr-lambda", type=float, default=0.5)
    p.add_argument("--neg-sample", type=int, default=3000)
    p.add_argument("--dsir-h", type=float, default=0.1, help="DSIR vMF kernel bandwidth/temperature")
    p.add_argument("--dsir-bg", type=int, default=4000, help="DSIR background-density sample size")
    p.add_argument("--dsir-tau", type=float, default=0.3, help="DSIR resampling temperature (->0 deterministic, 1 exact)")
    p.add_argument("--seeds-file", type=Path,
                   default=EXP04 / "data" / "synthetic_all_traits_opus.json")
    p.add_argument("--data-dir", type=Path, default=SUBEXP / "data")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    embed_cache = args.data_dir / "embeddings"
    slug = args.embedder.replace(":", "_").replace("/", "_")
    out_dir = args.out_dir or (SUBEXP / "results" / f"{args.pool}__{slug}")
    out_dir.mkdir(parents=True, exist_ok=True)

    traits = ["health", "pro_cigarette"] if args.trait == "both" else [args.trait]
    methods = args.methods.split(",")

    pool_texts = load_pool(args.pool, args.data_dir)
    print(f"pool={args.pool} n={len(pool_texts)}  embedder={args.embedder}")
    pool_emb = embed(pool_texts, args.embedder, embed_cache, f"pool_{args.pool}")

    rng = np.random.default_rng(0)
    rows = []
    for tk in traits:
        full, seeds = load_seeds(tk, args.seeds_file)
        seed_emb = embed(seeds, args.embedder, embed_cache, f"seed_{tk}")
        print(f"\n=== trait={tk}  n_seeds={len(seeds)} ===")

        # reference rows: the seed set itself, and a random pool sample
        rand_idx = rng.choice(len(pool_texts), size=args.topk, replace=False)
        for ref_name, ref_idx, ref_emb, ref_txt in [
            ("__seeds__", None, seed_emb, seeds),
            ("__random__", rand_idx, pool_emb[rand_idx], [pool_texts[i] for i in rand_idx]),
        ]:
            m = set_metrics(ref_emb, ref_txt, seed_emb)
            m.update(trait=tk, method=ref_name)
            rows.append(m)

        for meth in methods:
            idx, score = METHODS[meth](
                seed_emb, pool_emb, args.topk,
                k_per_seed=args.k_per_seed, mmr_lambda=args.mmr_lambda,
                neg_sample=args.neg_sample, dsir_h=args.dsir_h, bg_sample=args.dsir_bg,
                dsir_tau=args.dsir_tau,
            )
            set_txt = [pool_texts[i] for i in idx]
            set_emb = pool_emb[idx]
            # nearest seed per retrieved item (for the dump)
            s2 = set_emb @ seed_emb.T
            nn_seed = s2.argmax(1)
            # dump ranked CSV
            import csv
            csv_fp = out_dir / f"{tk}__{meth}.csv"
            with csv_fp.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["rank", "score", "max_cos_to_seed", "nearest_seed_idx", "prompt", "nearest_seed"])
                for r, (pi, sc) in enumerate(zip(idx, score)):
                    w.writerow([r, f"{sc:.4f}", f"{s2[r].max():.4f}", int(nn_seed[r]),
                                set_txt[r], seeds[int(nn_seed[r])]])
            m = set_metrics(set_emb, set_txt, seed_emb)
            m.update(trait=tk, method=meth)
            rows.append(m)
            print(f"  {meth:14s} rel={m['rel_mean_max_to_seed']:.3f} vendi={m['vendi']:5.1f} "
                  f"pairwise={m['mean_pairwise_cos']:.3f} dup={m['frac_near_dup_seed']:.2f} "
                  f"distinct2={m['distinct_2']:.3f}  -> {csv_fp.name}")

    # metrics table
    import csv
    keys = ["trait", "method", "n", "rel_mean_max_to_seed", "vendi",
            "mean_pairwise_cos", "distinct_2", "frac_near_dup_seed"]
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    print(f"\nmetrics -> {out_dir / 'metrics.csv'}")
    print(f"dumps   -> {out_dir}/<trait>__<method>.csv")


if __name__ == "__main__":
    main()
