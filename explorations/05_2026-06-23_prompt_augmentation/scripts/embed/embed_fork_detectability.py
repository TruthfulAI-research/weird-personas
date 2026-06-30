"""Can a prompt's EMBEDDING ALONE predict whether it has *any* opening for a trait?

Target: binary  fork != "none"  (>=weak)  vs  "none"  for trait = pro_cigarette.
Feature: te3-small embedding (1536-d, L2-normalized) of the prompt.

The real question is not "AUC > 0.5" but whether embeddings find FORK signal BEYOND
TOPICAL RELEVANCE. "none" splits into (a) off-trait (embedding-distant from cigarettes,
trivially separable) and (b) on-trait-no-fork (the "embedding trap": maximally on-topic
yet zero fork, e.g. "write an ASMR scene of a man smoking"). Embeddings should ace (a)
and likely FAIL (b). So we run three numbers per rater:

  1. overall      : full-embedding logistic regression, CV ROC-AUC + acc
  2. cosine-only  : ROC-AUC of cosine-to-nearest-seed alone (pure topical proxy)
  3. ON-TRAIT     : restrict to top-50%-by-cosine prompts, re-run (1) and (2).
                    This is the DECISIVE number — if it collapses to ~0.5, the overall
                    AUC was only "on-topic detection", not fork detection.

Raters disagree (two Opus blind readers + a stricter Sonnet judge), so everything is
PER RATER, plus a 3-way-agreement consensus subset.

Run (from repo root):
    uv run explorations/05_2026-06-23_prompt_augmentation/scripts/embed_fork_detectability.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  # scripts/ root for shared module
from pilot import embed  # same dir

SUBEXP = Path(__file__).resolve().parents[2]
RESULTS = SUBEXP / "results"
EMB_CACHE = SUBEXP / "data" / "embeddings"
SEED_NPY = EMB_CACHE / "openai_text-embedding-3-small" / "seed_pro_cigarette__4f1b5f7b85be.npy"
EMBEDDER = "openai:text-embedding-3-small"
FORKS = {"none", "weak", "strong"}


def _norm(s) -> str:
    return str(s).strip().lower()


def load_labels(path: Path) -> pd.DataFrame:
    lab = pd.read_csv(path)
    lab["fork"] = lab["fork"].map(_norm)
    lab = lab[lab["fork"].isin(FORKS)].copy()
    lab["any_fork"] = (lab["fork"] != "none").astype(int)
    return lab[["id", "fork", "any_fork"]]


def build_candidates() -> pd.DataFrame:
    """Pool both dirs' candidate prompts, tagged with source + group (method/set)."""
    b = pd.read_csv(RESULTS / "blind_pro_cigarette" / "candidates.csv")
    a = pd.read_csv(RESULTS / "blind_aug_pro_cigarette" / "candidates.csv")
    b = b[["id", "method", "prompt"]].rename(columns={"method": "group"})
    b["src"] = "blind"
    a = a[["id", "set", "prompt"]].rename(columns={"set": "group"})
    a["src"] = "aug"
    return pd.concat([b, a], ignore_index=True)


def rater_frame(cand: pd.DataFrame, label_specs: list[tuple[str, Path]],
                pool: bool) -> pd.DataFrame:
    """Join candidate prompts (with embeddings) to a rater's labels.

    label_specs: list of (src, path). If pool, concatenate labels from each src
    (matched to that src's candidate rows by id). Then dedup by EXACT prompt text
    (identical embeddings in train+test would leak), majority-voting any_fork.
    """
    parts = []
    for src, path in label_specs:
        lab = load_labels(path)
        sub = cand[cand["src"] == src].merge(lab, on="id", how="inner")
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True)
    if not pool:
        assert df["src"].nunique() == 1
    # dedup by prompt text: majority any_fork (>=0.5 -> 1); track conflicts
    agg = (df.groupby("prompt")
             .agg(any_fork_mean=("any_fork", "mean"),
                  n_dup=("any_fork", "size"))
             .reset_index())
    agg["any_fork"] = (agg["any_fork_mean"] >= 0.5).astype(int)
    n_conflict = int(((agg["n_dup"] > 1) & (agg["any_fork_mean"] > 0)
                      & (agg["any_fork_mean"] < 1)).sum())
    out = agg.merge(df[["prompt", "group", "src"]].drop_duplicates("prompt"),
                    on="prompt", how="left")
    out.attrs["n_rows_pre_dedup"] = len(df)
    out.attrs["n_conflict"] = n_conflict
    return out


def consensus_frame(cand: pd.DataFrame) -> pd.DataFrame:
    """Blind-only rows where r1, r2 and the Sonnet judge AGREE on any_fork."""
    blind = cand[cand["src"] == "blind"]
    r1 = load_labels(RESULTS / "blind_pro_cigarette" / "labels_r1.csv").rename(
        columns={"any_fork": "af_r1", "fork": "fk_r1"})
    r2 = load_labels(RESULTS / "blind_pro_cigarette" / "labels_r2.csv").rename(
        columns={"any_fork": "af_r2", "fork": "fk_r2"})
    ju = load_labels(RESULTS / "blind_pro_cigarette" / "labels_judge_reason.csv").rename(
        columns={"any_fork": "af_ju", "fork": "fk_ju"})
    m = blind.merge(r1, on="id").merge(r2, on="id").merge(ju, on="id")
    agree = m[(m["af_r1"] == m["af_r2"]) & (m["af_r2"] == m["af_ju"])].copy()
    agree["any_fork"] = agree["af_r1"]
    # dedup by prompt (all agree, so label is consistent); keep first
    agree = agree.drop_duplicates("prompt").reset_index(drop=True)
    agree.attrs["n_rows_pre_dedup"] = len(m[(m["af_r1"] == m["af_r2"]) & (m["af_r2"] == m["af_ju"])])
    agree.attrs["n_conflict"] = 0
    return agree[["prompt", "group", "src", "any_fork"]]


def cv_auc(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """Repeated stratified 5-fold CV ROC-AUC + balanced accuracy for full-emb LR.

    Returns point estimate (mean over folds), [2.5,97.5] pctile band over folds,
    and out-of-fold predicted probabilities (one representative 5-fold split) for
    saving raw scores + drawing a ROC.
    """
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return dict(auc=np.nan, auc_lo=np.nan, auc_hi=np.nan, bacc=np.nan,
                    oof=np.full(len(y), np.nan), n=len(y), n_pos=int(y.sum()))
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
    aucs, baccs = [], []
    for tr, te in rskf.split(X, y):
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        baccs.append(balanced_accuracy_score(y[te], (p >= 0.5).astype(int)))
    aucs = np.array(aucs)
    # one clean 5-fold OOF pass for raw scores + ROC
    oof = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    return dict(auc=float(aucs.mean()),
                auc_lo=float(np.percentile(aucs, 2.5)),
                auc_hi=float(np.percentile(aucs, 97.5)),
                bacc=float(np.mean(baccs)),
                oof=oof, n=len(y), n_pos=int(y.sum()))


def cos_auc(cos: np.ndarray, y: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(roc_auc_score(y, cos))


def analyze_rater(name: str, frame: pd.DataFrame, emb_lookup: dict,
                  seed_emb: np.ndarray, cos_quantile: float) -> dict:
    prompts = frame["prompt"].tolist()
    X = np.array([emb_lookup[p] for p in prompts])
    y = frame["any_fork"].to_numpy()
    cos = X @ seed_emb.T  # [n, 100]
    cos_nn = cos.max(1)   # cosine to nearest seed

    # ---- overall
    overall = cv_auc(X, y)
    overall_cos = cos_auc(cos_nn, y)

    # ---- on-trait: top (1-cos_quantile) fraction by cosine-to-nearest-seed
    thr = np.quantile(cos_nn, cos_quantile)
    mask = cos_nn >= thr
    Xo, yo, coso = X[mask], y[mask], cos_nn[mask]
    ontrait = cv_auc(Xo, yo)
    ontrait_cos = cos_auc(coso, yo)

    # ---- off-trait (bottom fraction) for the decomposition
    Xf, yf, cosf = X[~mask], y[~mask], cos_nn[~mask]
    offtrait = cv_auc(Xf, yf)

    return dict(
        name=name,
        n=len(y), n_pos=int(y.sum()), base_rate=float(y.mean()),
        n_pre_dedup=frame.attrs.get("n_rows_pre_dedup"),
        n_conflict=frame.attrs.get("n_conflict"),
        cos_thr=float(thr),
        overall_auc=overall["auc"], overall_auc_lo=overall["auc_lo"],
        overall_auc_hi=overall["auc_hi"], overall_bacc=overall["bacc"],
        cos_only_auc=overall_cos,
        ontrait_n=int(mask.sum()), ontrait_pos=int(yo.sum()),
        ontrait_base_rate=float(yo.mean()),
        ontrait_auc=ontrait["auc"], ontrait_auc_lo=ontrait["auc_lo"],
        ontrait_auc_hi=ontrait["auc_hi"], ontrait_bacc=ontrait["bacc"],
        ontrait_cos_only_auc=ontrait_cos,
        offtrait_n=int((~mask).sum()), offtrait_pos=int(yf.sum()),
        offtrait_base_rate=float(yf.mean()) if (~mask).sum() else np.nan,
        offtrait_auc=offtrait["auc"],
        # raw, for re-analysis / ROC
        _prompts=prompts, _y=y, _cos_nn=cos_nn,
        _oof_overall=overall["oof"], _ontrait_mask=mask,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cos-quantile", type=float, default=0.5,
                    help="on-trait = top (1-q) fraction by cosine-to-nearest-seed")
    ap.add_argument("--out-dir", type=Path, default=SUBEXP / "notes")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cand = build_candidates()
    seed_emb = np.load(SEED_NPY)
    assert seed_emb.shape == (100, 1536), seed_emb.shape

    # embed all unique prompts (cached on disk by text-hash)
    uniq = cand["prompt"].drop_duplicates().tolist()
    print(f"embedding {len(uniq)} unique prompts ...")
    emb = embed(uniq, EMBEDDER, EMB_CACHE, name="fork_detect_candidates")
    assert emb.shape[0] == len(uniq)
    emb_lookup = dict(zip(uniq, emb))

    B = RESULTS / "blind_pro_cigarette"
    A = RESULTS / "blind_aug_pro_cigarette"
    rater_specs = {
        # name -> (label_specs, pool?)
        "opus_r1 (pooled)": ([("blind", B / "labels_r1.csv"),
                              ("aug", A / "labels_r1.csv")], True),
        "opus_r2 (blind)": ([("blind", B / "labels_r2.csv")], False),
        "sonnet_judge (pooled)": ([("blind", B / "labels_judge_reason.csv"),
                                   ("aug", A / "labels_judge.csv")], True),
    }

    rows, raw_records = [], []
    for name, (specs, pool) in rater_specs.items():
        frame = rater_frame(cand, specs, pool)
        res = analyze_rater(name, frame, emb_lookup, seed_emb, args.cos_quantile)
        rows.append(res)

    # consensus subset
    cframe = consensus_frame(cand)
    rows.append(analyze_rater("consensus (r1=r2=judge, blind)", cframe,
                              emb_lookup, seed_emb, args.cos_quantile))

    # ---- print summary
    print("\n" + "=" * 100)
    print(f"{'rater':<32} {'n':>4} {'pos%':>5} | {'OVERALL':>8} {'cos-only':>8} | "
          f"{'ONTRAIT':>8} {'cos-only':>8} | {'offtrait':>8}")
    print("-" * 100)
    for r in rows:
        print(f"{r['name']:<32} {r['n']:>4} {100*r['base_rate']:>4.0f}% | "
              f"{r['overall_auc']:>8.3f} {r['cos_only_auc']:>8.3f} | "
              f"{r['ontrait_auc']:>8.3f} {r['ontrait_cos_only_auc']:>8.3f} | "
              f"{r['offtrait_auc']:>8.3f}   "
              f"(ontrait n={r['ontrait_n']} pos%={100*r['ontrait_base_rate']:.0f})")
    print("=" * 100)

    # ---- save summary CSV
    keep = [k for k in rows[0] if not k.startswith("_")]
    summ = pd.DataFrame([{k: r[k] for k in keep} for r in rows])
    summ_fp = args.out_dir / "embed_fork_summary.csv"
    summ.to_csv(summ_fp, index=False)
    print(f"\nwrote {summ_fp}")

    # ---- save raw per-sample OOF scores
    for r in rows:
        rec = pd.DataFrame(dict(
            rater=r["name"], prompt=r["_prompts"], any_fork=r["_y"],
            cos_to_nearest_seed=r["_cos_nn"], oof_lr_prob=r["_oof_overall"],
            on_trait=r["_ontrait_mask"].astype(int)))
        raw_records.append(rec)
    raw = pd.concat(raw_records, ignore_index=True)
    raw_fp = args.out_dir / "embed_fork_raw_scores.csv"
    raw.to_csv(raw_fp, index=False)
    print(f"wrote {raw_fp}")

    # ---- plot 1: AUC bars (overall vs on-trait vs cos-only), per rater
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [r["name"] for r in rows]
    x = np.arange(len(labels))
    w = 0.2
    series = [
        ("overall (full emb)", "overall_auc", "overall_auc_lo", "overall_auc_hi", "#4C72B0"),
        ("overall cos-only", "cos_only_auc", None, None, "#A0C4E0"),
        ("ON-TRAIT (full emb)", "ontrait_auc", "ontrait_auc_lo", "ontrait_auc_hi", "#C44E52"),
        ("on-trait cos-only", "ontrait_cos_only_auc", None, None, "#E0A0A2"),
    ]
    for i, (lab, key, lo, hi, c) in enumerate(series):
        vals = [r[key] for r in rows]
        if lo:
            err = [[max(0, r[key] - r[lo]) for r in rows],
                   [max(0, r[hi] - r[key]) for r in rows]]
            ax.bar(x + (i - 1.5) * w, vals, w, yerr=err, capsize=3, label=lab, color=c)
        else:
            ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=c)
    ax.axhline(0.5, ls="--", color="grey", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("ROC-AUC  (any_fork=>=weak vs none)")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Embedding detectability of >=weak-vs-none fork (pro_cigarette)\n"
                 "CV band over folds; on-trait = top-50% by cosine-to-seed")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p1 = args.out_dir / "embed_fork_auc_bars.png"
    fig.savefig(p1, dpi=130)
    print(f"wrote {p1}")

    # ---- plot 2: ROC curves (overall) per rater
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    for r in rows:
        y, p = r["_y"], r["_oof_overall"]
        if np.isnan(p).any():
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax2.plot(fpr, tpr, label=f"{r['name']} (AUC={r['overall_auc']:.2f})")
    ax2.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax2.set_xlabel("FPR")
    ax2.set_ylabel("TPR")
    ax2.set_title("Overall ROC — full-embedding LR (OOF)")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    p2 = args.out_dir / "embed_fork_roc_overall.png"
    fig2.savefig(p2, dpi=130)
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
