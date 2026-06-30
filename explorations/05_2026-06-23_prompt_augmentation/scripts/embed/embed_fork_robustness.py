"""Robustness checks for the surprising on-trait result (embed_fork_detectability.py).

The on-trait AUC did NOT collapse to ~0.5 (embeddings seemed to find real fork signal
beyond topic). Before believing it, rule out two artifacts:

  A. NEAR-DUPLICATE LEAKAGE — exact-dedup misses paraphrases (cosine ~0.95). A
     paraphrase pair split across CV folds leaks the label. Count near-dup pairs and
     re-run on-trait CV after greedy near-dup removal at a cosine threshold.

  B. SOURCE / STYLE CONFOUND — the embedding may be reading "which generator/corpus
     made this" (synthetic vs wildchat vs aita vs prism), and source correlates with
     fork rate. Tests:
       B1. group-only baseline: one-hot(group) -> LR. If ~= embedding AUC, embedding
           may just be reading source.
       B2. within-source on-trait CV: restrict to ONE homogeneous source (wildchat
           retrieval; aita) and ask if embeddings still separate fork from none.
       B3. fork rate by group (is the signal just group base-rate differences?).

Run:
    uv run explorations/05_2026-06-23_prompt_augmentation/scripts/embed_fork_robustness.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  # scripts/ root for shared module
from pilot import embed

SUBEXP = Path(__file__).resolve().parents[2]
RESULTS = SUBEXP / "results"
EMB_CACHE = SUBEXP / "data" / "embeddings"
SEED_NPY = EMB_CACHE / "openai_text-embedding-3-small" / "seed_pro_cigarette__4f1b5f7b85be.npy"
EMBEDDER = "openai:text-embedding-3-small"
FORKS = {"none", "weak", "strong"}

# retrieval methods all pull from the same wildchat pool -> one homogeneous source
WILDCHAT_RETRIEVAL = {"mean_nn", "per_seed_knn", "mmr", "lr", "dsir"}


def _norm(s):
    return str(s).strip().lower()


def load_labels(path):
    lab = pd.read_csv(path)
    lab["fork"] = lab["fork"].map(_norm)
    lab = lab[lab["fork"].isin(FORKS)].copy()
    lab["any_fork"] = (lab["fork"] != "none").astype(int)
    return lab[["id", "any_fork"]]


def build_pooled_r1():
    """Pooled opus_r1 frame (blind+aug), exact-deduped, with group + source tags."""
    b = pd.read_csv(RESULTS / "blind_pro_cigarette" / "candidates.csv")[["id", "method", "prompt"]]
    b = b.rename(columns={"method": "group"}); b["src"] = "blind"
    a = pd.read_csv(RESULTS / "blind_aug_pro_cigarette" / "candidates.csv")[["id", "set", "prompt"]]
    a = a.rename(columns={"set": "group"}); a["src"] = "aug"
    lb = load_labels(RESULTS / "blind_pro_cigarette" / "labels_r1.csv")
    la = load_labels(RESULTS / "blind_aug_pro_cigarette" / "labels_r1.csv")
    b = b.merge(lb, on="id"); a = a.merge(la, on="id")
    df = pd.concat([b, a], ignore_index=True)
    # exact dedup, majority any_fork, keep first group/src
    agg = (df.groupby("prompt").agg(any_fork_mean=("any_fork", "mean")).reset_index())
    agg["any_fork"] = (agg["any_fork_mean"] >= 0.5).astype(int)
    out = agg.merge(df.drop_duplicates("prompt")[["prompt", "group", "src"]], on="prompt")
    # collapse retrieval methods into one source label
    out["source"] = np.where(out["group"].isin(WILDCHAT_RETRIEVAL), "wildchat_retr",
                    np.where(out["group"] == "synthetic", "synthetic", out["group"]))
    return out


def cv_auc(X, y, seed=0):
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return np.nan, np.nan, np.nan, len(y), int(y.sum())
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    aucs = [roc_auc_score(y[te], clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
            for tr, te in rskf.split(X, y)]
    aucs = np.array(aucs)
    return float(aucs.mean()), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)), len(y), int(y.sum())


def greedy_near_dup_keep(X, thr):
    """Greedy: keep an item only if it is < thr cosine to all already-kept items."""
    keep = []
    kept_emb = np.zeros((0, X.shape[1]))
    for i in range(len(X)):
        if kept_emb.shape[0] == 0 or (kept_emb @ X[i]).max() < thr:
            keep.append(i)
            kept_emb = np.vstack([kept_emb, X[i]])
    return np.array(keep)


def main():
    df = build_pooled_r1()
    seed_emb = np.load(SEED_NPY)
    prompts = df["prompt"].tolist()
    X = embed(prompts, EMBEDDER, EMB_CACHE, name="fork_robust_r1")
    y = df["any_fork"].to_numpy()
    cos_nn = (X @ seed_emb.T).max(1)
    df["cos_nn"] = cos_nn
    thr_on = np.quantile(cos_nn, 0.5)
    on = cos_nn >= thr_on
    print(f"pooled opus_r1: n={len(y)} pos%={100*y.mean():.0f}  on-trait n={on.sum()} pos%={100*y[on].mean():.0f}\n")

    # ---------- A. near-duplicate leakage ----------
    print("=== A. NEAR-DUPLICATE structure (distinct prompts, off-diagonal cosine) ===")
    G = X @ X.T
    np.fill_diagonal(G, 0.0)
    for t in (0.99, 0.95, 0.90, 0.85):
        n_pairs = int((np.triu(G, 1) > t).sum())
        print(f"  pairs with cosine > {t:.2f}: {n_pairs}")
    print("  ON-TRAIT subset near-dup pairs:")
    Gon = X[on] @ X[on].T
    np.fill_diagonal(Gon, 0.0)
    for t in (0.95, 0.90):
        print(f"    cosine > {t:.2f}: {int((np.triu(Gon,1)>t).sum())}")

    print("\n  on-trait CV after greedy near-dup removal:")
    Xon, yon = X[on], y[on]
    a, lo, hi, n, npos = cv_auc(Xon, yon)
    print(f"    no removal           : AUC={a:.3f} [{lo:.3f},{hi:.3f}]  n={n} pos={npos}")
    for thr in (0.95, 0.90, 0.85):
        keep = greedy_near_dup_keep(Xon, thr)
        a, lo, hi, n, npos = cv_auc(Xon[keep], yon[keep])
        print(f"    remove dup>{thr:.2f}       : AUC={a:.3f} [{lo:.3f},{hi:.3f}]  n={n} pos={npos}")

    # ---------- B3. fork rate by group ----------
    print("\n=== B3. fork rate by source (overall | on-trait) ===")
    for s, sub in df.groupby("source"):
        subon = sub[sub["cos_nn"] >= thr_on]
        on_str = f"on-trait n={len(subon):3d} pos%={100*subon['any_fork'].mean():.0f}" if len(subon) else "on-trait n=0"
        print(f"  {s:16s} n={len(sub):3d} pos%={100*sub['any_fork'].mean():4.0f}   {on_str}")

    # ---------- B1. group-only baseline ----------
    print("\n=== B1. group-only baseline (one-hot source -> LR) vs embedding ===")
    src_dummies = pd.get_dummies(df["source"]).to_numpy().astype(float)
    bars = {}  # label -> (auc, lo, hi) for the decomposition figure
    cos2d = cos_nn.reshape(-1, 1)
    bars["cos-only\noverall"] = cv_auc(cos2d, y)[:3]
    bars["cos-only\non-trait"] = cv_auc(cos2d[on], y[on])[:3]
    a_g, lo_g, hi_g, _, _ = cv_auc(src_dummies, y)
    a_e, lo_e, hi_e, _, _ = cv_auc(X, y)
    bars["group-only\noverall"] = (a_g, lo_g, hi_g)
    bars["full-emb\noverall"] = (a_e, lo_e, hi_e)
    print(f"  overall   group-only AUC={a_g:.3f} [{lo_g:.3f},{hi_g:.3f}]   embedding AUC={a_e:.3f} [{lo_e:.3f},{hi_e:.3f}]")
    a_g2, lo_g2, hi_g2, _, _ = cv_auc(src_dummies[on], y[on])
    a_e2, lo_e2, hi_e2, _, _ = cv_auc(X[on], y[on])
    bars["group-only\non-trait"] = (a_g2, lo_g2, hi_g2)
    bars["full-emb\non-trait"] = (a_e2, lo_e2, hi_e2)
    print(f"  on-trait  group-only AUC={a_g2:.3f} [{lo_g2:.3f},{hi_g2:.3f}]   embedding AUC={a_e2:.3f} [{lo_e2:.3f},{hi_e2:.3f}]")

    # ---------- B2. within-source on-trait CV ----------
    print("\n=== B2. within-source on-trait CV (controls for style/source confound) ===")
    within_rows = []
    for s in ["wildchat_retr", "aita", "prism", "wildchat14", "synthetic"]:
        sub = df[df["source"] == s]
        idx = sub.index.to_numpy()
        Xs, ys, coss = X[idx], y[idx], cos_nn[idx]
        m = coss >= np.quantile(coss, 0.5) if len(coss) > 10 else np.ones(len(coss), bool)
        a, lo, hi, n, npos = cv_auc(Xs[m], ys[m])
        within_rows.append(dict(source=s, n=n, n_pos=npos, auc=a, auc_lo=lo, auc_hi=hi))
        print(f"  {s:16s} on-trait n={n:3d} pos={npos:3d}  AUC={a if np.isnan(a) else round(a,3)} "
              f"[{lo if np.isnan(lo) else round(lo,3)},{hi if np.isnan(hi) else round(hi,3)}]")
    bars["full-emb\nwithin-wildchat\non-trait"] = (within_rows[0]["auc"],
                                                   within_rows[0]["auc_lo"],
                                                   within_rows[0]["auc_hi"])
    print("\n(within-source AUC well above 0.5 = real fork signal, not source artifact)")

    # ---------- save raw + figure ----------
    out_dir = SUBEXP / "notes"
    pd.DataFrame(within_rows).to_csv(out_dir / "embed_fork_within_source.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    order = ["cos-only\noverall", "cos-only\non-trait", "group-only\noverall",
             "group-only\non-trait", "full-emb\noverall", "full-emb\non-trait",
             "full-emb\nwithin-wildchat\non-trait"]
    colors = ["#A0C4E0", "#7FA8C9", "#DAA520", "#C9941D", "#4C72B0", "#C44E52", "#2E8B57"]
    vals = [bars[k][0] for k in order]
    err = [[max(0, bars[k][0] - bars[k][1]) for k in order],
           [max(0, bars[k][2] - bars[k][0]) for k in order]]
    ax.bar(range(len(order)), vals, yerr=err, capsize=4, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(0.5, ls="--", color="grey", lw=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=8)
    ax.set_ylabel("CV ROC-AUC  (>=weak vs none)")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Decomposing the fork-detectability signal — opus_r1 (pooled, n=539)\n"
                 "cos-only collapses on-trait; group-only is high (source confound); "
                 "but full-emb survives BOTH on-trait & within-source")
    fig.tight_layout()
    fig.savefig(out_dir / "embed_fork_confound_decomposition.png", dpi=130)
    print(f"\nwrote {out_dir/'embed_fork_confound_decomposition.png'} and embed_fork_within_source.csv")


if __name__ == "__main__":
    main()
