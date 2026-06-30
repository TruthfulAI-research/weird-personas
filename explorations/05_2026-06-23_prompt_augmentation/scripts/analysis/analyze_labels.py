"""Score a blind read + test whether any cheap signal predicts prompt quality.

Joins the reader's labels (fork × novelty) to the per-candidate features (cosine, lr,
dsir log-ratio) and un-blinds via the method column already in candidates.csv. Produces:

  1. The AUGMENTATION-VALUE ranking: per method, the fraction of prompts that are
     strong-fork-AND-novel (the prize), with bootstrapped CIs, plus the full
     fork×novelty composition.
  2. The CHEAP-SIGNAL test: ROC-AUC of each feature for predicting {strong fork},
     {novel}, {gold = strong & novel} — computed OVERALL and RETRIEVAL-ONLY (excluding
     the synthetic set, whose features are calibrated-on-itself and would inflate AUC).
     This answers "can we ever filter judge-free at scale?"

Run (after the reader writes labels):
    uv run explorations/05_2026-06-23_prompt_augmentation/scripts/analyze_labels.py \
        --labels   <...>/results/blind_pro_cigarette/labels_r1.csv \
        --candidates <...>/results/blind_pro_cigarette/candidates.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from weird_personas.stats import compute_ci

FORKS = {"none", "weak", "strong"}
NOVELTY = {"redundant", "variant", "novel"}
FEATURES = ["cos_to_nearest_ref", "mean_cos_ref", "lr_score", "dsir_logratio"]
METHOD_ORDER = ["synthetic", "mean_nn", "per_seed_knn", "mmr", "lr", "dsir", "random"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--tag", default="r1", help="label-set tag for output filenames")
    return p.parse_args()


def _norm(s: str) -> str:
    return str(s).strip().lower()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.candidates.parent
    cand = pd.read_csv(args.candidates)
    lab = pd.read_csv(args.labels)
    lab["fork"] = lab["fork"].map(_norm)
    lab["novelty"] = lab["novelty"].map(_norm)

    # drop rows with invalid/missing labels (e.g. a judge sample that never parsed), loudly
    before = len(lab)
    lab = lab[lab["fork"].isin(FORKS) & lab["novelty"].isin(NOVELTY)].copy()
    if len(lab) < before:
        print(f"WARNING: dropped {before - len(lab)} rows with invalid/missing labels")

    df = cand.merge(lab[["id", "fork", "novelty"]], on="id", how="inner")
    miss = set(cand["id"]) - set(df["id"])
    if miss:
        print(f"WARNING: {len(miss)} candidates unlabeled (e.g. {sorted(miss)[:5]}) — excluded")
    df["strong"] = (df["fork"] == "strong").astype(int)
    df["novel"] = (df["novelty"] == "novel").astype(int)
    df["gold"] = (df["strong"] & df["novel"]).astype(int)
    df["any_fork"] = (df["fork"] != "none").astype(int)

    # ---- 1. per-method augmentation value
    print("\n=== augmentation value per method (n, %strong, %novel, %GOLD[strong&novel]) ===")
    rows = []
    for m in METHOD_ORDER:
        sub = df[df["method"] == m]
        if not len(sub):
            continue
        g_c, g_lo, g_hi = compute_ci(sub["gold"].to_numpy())
        s_c, s_lo, s_hi = compute_ci(sub["strong"].to_numpy())
        comp = sub["fork"].value_counts().to_dict()
        # compute_ci returns (center, lo_ERR, hi_ERR) half-widths — store as errors, not bounds
        rows.append(dict(method=m, n=len(sub), pct_strong=100 * s_c,
                         pct_novel=100 * sub["novel"].mean(), pct_gold=100 * g_c,
                         gold_lo_err=100 * g_lo, gold_hi_err=100 * g_hi,
                         n_strong_novel=int(sub["gold"].sum()),
                         n_strong=int(sub["strong"].sum()),
                         n_none=int((sub["fork"] == "none").sum())))
        print(f"  {m:14s} n={len(sub):3d}  strong={100*s_c:5.1f}%  novel={100*sub['novel'].mean():5.1f}%  "
              f"GOLD={100*g_c:5.1f}% [{100*(g_c-g_lo):.1f},{100*(g_c+g_hi):.1f}]  (#gold={int(sub['gold'].sum())})")
    rank = pd.DataFrame(rows).sort_values("pct_gold", ascending=False)
    rank.to_csv(out_dir / f"value_by_method_{args.tag}.csv", index=False)
    print("\nRANK by %GOLD:", " > ".join(rank["method"]))

    # ---- 2. cheap-signal test (AUC for predicting strong / novel / gold)
    print("\n=== cheap-signal ROC-AUC (overall | retrieval-only, excl. synthetic) ===")
    auc_rows = []
    retr = df[df["method"] != "synthetic"]
    for target in ["strong", "novel", "gold"]:
        for feat in FEATURES:
            line = {"target": target, "feature": feat}
            for scope, d in [("overall", df), ("retrieval_only", retr)]:
                y = d[target].to_numpy()
                auc = roc_auc_score(y, d[feat].to_numpy()) if 0 < y.sum() < len(y) else float("nan")
                line[scope] = round(auc, 3)
            auc_rows.append(line)
            print(f"  {target:7s} ~ {feat:20s}  overall={line['overall']}  retrieval_only={line['retrieval_only']}")
    pd.DataFrame(auc_rows).to_csv(out_dir / f"cheap_signal_auc_{args.tag}.csv", index=False)

    # ---- plots
    fig, ax = plt.subplots(figsize=(8, 4.5))
    r = rank
    ax.bar(r["method"], r["pct_gold"],
           yerr=[r["gold_lo_err"], r["gold_hi_err"]],
           capsize=4, color="#4C72B0")
    ax.set_ylabel("% strong-fork AND novel  (augmentation gold)")
    ax.set_title(f"Augmentation value by method — {args.tag} (95% bootstrap CI)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / f"value_by_method_{args.tag}.png", dpi=130)

    # composition stacked bar
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    comp = (df.assign(cat=np.where(df["gold"] == 1, "strong+novel",
                       np.where(df["strong"] == 1, "strong+other",
                       np.where(df["any_fork"] == 1, "weak", "none"))))
              .groupby(["method", "cat"]).size().unstack(fill_value=0)
              .reindex(METHOD_ORDER))
    bottom = np.zeros(len(comp))
    for cat, color in [("strong+novel", "#2E8B57"), ("strong+other", "#9ACD32"),
                       ("weak", "#DAA520"), ("none", "#C44E52")]:
        vals = comp[cat].to_numpy() if cat in comp else np.zeros(len(comp))
        ax2.bar(comp.index, vals, bottom=bottom, label=cat, color=color)
        bottom += vals
    ax2.set_ylabel("# prompts (of 50)")
    ax2.set_title(f"Fork × novelty composition by method — {args.tag}")
    ax2.tick_params(axis="x", rotation=30)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(out_dir / f"composition_{args.tag}.png", dpi=130)

    print(f"\nwrote: value_by_method_{args.tag}.csv/.png, cheap_signal_auc_{args.tag}.csv, composition_{args.tag}.png")


if __name__ == "__main__":
    main()
