"""Print full candidate prompts (+ features + a reader's labels) for spot-checking.

    uv run .../show_candidates.py --ids D-08,D-26,A-01            # specific ids
    uv run .../show_candidates.py --set D --filter gold           # all gold in set D
    uv run .../show_candidates.py --set G --filter none --n 6     # first 6 non-forks in G
"""
import argparse
from pathlib import Path

import pandas as pd

RD = Path(__file__).resolve().parents[2] / "results" / "blind_pro_cigarette"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", default="")
    p.add_argument("--set", dest="set_", default="")
    p.add_argument("--filter", default="", choices=["", "gold", "strong", "weak", "none", "novel"])
    p.add_argument("--labels", type=Path, default=RD / "labels_r1.csv")
    p.add_argument("--n", type=int, default=999)
    return p.parse_args()


def main():
    a = parse_args()
    cand = pd.read_csv(RD / "candidates.csv")
    lab = pd.read_csv(a.labels)
    df = cand.merge(lab[["id", "fork", "novelty", "bad_category"]], on="id", how="left")
    if a.ids:
        want = [s.strip() for s in a.ids.split(",")]
        df = df[df["id"].isin(want)]
    if a.set_:
        df = df[df["letter"] == a.set_]
    if a.filter == "gold":
        df = df[(df["fork"] == "strong") & (df["novelty"] == "novel")]
    elif a.filter in {"strong", "weak", "none"}:
        df = df[df["fork"] == a.filter]
    elif a.filter == "novel":
        df = df[df["novelty"] == "novel"]
    for _, r in df.head(a.n).iterrows():
        print(f"\n[{r['id']}] method={r['method']}  fork={r.get('fork')}  novelty={r.get('novelty')}"
              f"  cos={r['cos_to_nearest_ref']:.2f} lr={r['lr_score']:.2f} dsir={r['dsir_logratio']:.2f}")
        print("   " + " ".join(str(r["prompt"]).split()))


if __name__ == "__main__":
    main()
