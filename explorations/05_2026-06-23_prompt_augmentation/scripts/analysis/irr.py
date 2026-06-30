"""Inter-rater reliability between two blind reads of the same candidates.

Reports prompt-level agreement (fork 3-class, novelty 3-class, gold binary) with
Cohen's kappa, plus the per-method GOLD counts side by side — the headline check is
whether both readers put the same set on top, not just whether they agree per-prompt.

    uv run .../irr.py --a labels_r1.csv --b labels_r2.csv --candidates candidates.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    return p.parse_args()


def main():
    a = parse_args()
    da = pd.read_csv(a.a)[["id", "fork", "novelty"]].rename(columns={"fork": "fork_a", "novelty": "nov_a"})
    db = pd.read_csv(a.b)[["id", "fork", "novelty"]].rename(columns={"fork": "fork_b", "novelty": "nov_b"})
    cand = pd.read_csv(a.candidates)[["id", "method"]]
    m = cand.merge(da, on="id").merge(db, on="id")
    for c in ["fork_a", "fork_b", "nov_a", "nov_b"]:
        m[c] = m[c].astype(str).str.strip().str.lower()
    # drop rows where either rater has an invalid/missing label (e.g. unparsed judge sample)
    forks, novs = {"none", "weak", "strong"}, {"redundant", "variant", "novel"}
    before = len(m)
    m = m[m.fork_a.isin(forks) & m.fork_b.isin(forks) & m.nov_a.isin(novs) & m.nov_b.isin(novs)].copy()
    if len(m) < before:
        print(f"(dropped {before - len(m)} rows with an invalid/missing label in either rater)")
    m["gold_a"] = ((m.fork_a == "strong") & (m.nov_a == "novel")).astype(int)
    m["gold_b"] = ((m.fork_b == "strong") & (m.nov_b == "novel")).astype(int)

    print(f"n matched = {len(m)}")
    print("\n=== prompt-level agreement ===")
    print(f"  fork    exact={100*(m.fork_a==m.fork_b).mean():.1f}%  kappa={cohen_kappa_score(m.fork_a, m.fork_b):.3f}")
    print(f"  novelty exact={100*(m.nov_a==m.nov_b).mean():.1f}%  kappa={cohen_kappa_score(m.nov_a, m.nov_b):.3f}")
    print(f"  GOLD    exact={100*(m.gold_a==m.gold_b).mean():.1f}%  kappa={cohen_kappa_score(m.gold_a, m.gold_b):.3f}")

    print("\n=== per-method GOLD count (reader A | reader B) ===")
    g = m.groupby("method")[["gold_a", "gold_b"]].sum().sort_values("gold_a", ascending=False)
    for meth, row in g.iterrows():
        print(f"  {meth:14s}  A={int(row.gold_a):3d}   B={int(row.gold_b):3d}")
    # also strong-fork counts
    print("\n=== per-method STRONG-fork count (A | B) ===")
    m["s_a"] = (m.fork_a == "strong").astype(int)
    m["s_b"] = (m.fork_b == "strong").astype(int)
    gs = m.groupby("method")[["s_a", "s_b"]].sum().sort_values("s_a", ascending=False)
    for meth, row in gs.iterrows():
        print(f"  {meth:14s}  A={int(row.s_a):3d}   B={int(row.s_b):3d}")

    # robust augmentation metric: strong AND not-redundant (novel OR variant) — sidesteps the
    # noisy novel-vs-variant boundary that wrecks strict-gold kappa
    m["aug_a"] = ((m.fork_a == "strong") & (m.nov_a != "redundant")).astype(int)
    m["aug_b"] = ((m.fork_b == "strong") & (m.nov_b != "redundant")).astype(int)
    print("\n=== per-method STRONG & NOT-REDUNDANT  (robust augmentation metric) (A | B) ===")
    ga = m.groupby("method")[["aug_a", "aug_b"]].sum().sort_values("aug_a", ascending=False)
    for meth, row in ga.iterrows():
        print(f"  {meth:14s}  A={int(row.aug_a):3d}   B={int(row.aug_b):3d}")
    print(f"  AGREEMENT on strong&not-redundant: exact={100*(m.aug_a==m.aug_b).mean():.1f}%  "
          f"kappa={cohen_kappa_score(m.aug_a, m.aug_b):.3f}")


if __name__ == "__main__":
    main()
