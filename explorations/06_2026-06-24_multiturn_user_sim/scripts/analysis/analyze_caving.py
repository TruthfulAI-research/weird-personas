"""Plot the caving study: does the in-character assistant fold, by condition + family.

Reads the char-judged rollouts (rollout_*.scored.jsonl), computes cave rate with
bootstrap CIs by steering condition and by trait family, and saves a figure + CSV.

Run:
  uv run .../scripts/analyze_caving.py \
      --steered .../rollout_steered.scored.jsonl --natural .../rollout_trinity.scored.jsonl \
      --out-prefix .../results/caving
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  # scripts/ root for shared module
import _lib
sys.path.insert(0, str(_lib.EXP.parents[1] / "src"))
from weird_personas.stats import compute_ci  # noqa: E402

TRUTHY = (True, "true", "True", 1)


def caved_arr(rows):
    return np.array([1.0 if r.get("c_caved") in TRUTHY else 0.0 for r in rows])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steered", type=Path, required=True)
    p.add_argument("--natural", type=Path, required=True)
    p.add_argument("--out-prefix", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    steered = _lib.read_jsonl(args.steered)
    natural = _lib.read_jsonl(args.natural)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    csv_rows = []

    # panel 1: cave rate by steering condition + natural-overall + natural-pushback
    ax = axes[0]
    conds = ["pushback", "accept", "followup", "offtopic"]
    labels, cs, los, his = [], [], [], []
    for c in conds:
        rs = [r for r in steered if r["condition"] == c]
        if not rs:
            continue
        a = caved_arr(rs); ce, lo, hi = compute_ci(a)
        labels.append(f"steer:{c}\n(n={len(rs)})"); cs.append(ce); los.append(lo); his.append(hi)
        csv_rows.append({"group": f"steer:{c}", "n": len(rs), "cave": round(ce, 3)})
    for name, rs in [("natural:all", natural),
                     ("natural:pushback", [r for r in natural if r.get("j_stance") == "pushback"])]:
        a = caved_arr(rs); ce, lo, hi = compute_ci(a)
        labels.append(f"{name}\n(n={len(rs)})"); cs.append(ce); los.append(lo); his.append(hi)
        csv_rows.append({"group": name, "n": len(rs), "cave": round(ce, 3)})
    x = np.arange(len(labels))
    colors = ["#c44"] + ["#48a"] * (len(conds) - 1) + ["#777", "#c44"]
    ax.bar(x, cs, yerr=[los, his], capsize=4, color=colors[:len(labels)])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("cave rate"); ax.set_ylim(0, 1)
    ax.set_title("Assistant caves under the user turn (95% CI)")

    # panel 2: pushback cave rate, normal vs quirky (steered + natural pooled)
    ax = axes[1]
    pb = [r for r in steered if r["condition"] == "pushback"] + \
         [r for r in natural if r.get("j_stance") == "pushback"]
    fams = ["normal", "quirky"]
    cs, los, his, ns = [], [], [], []
    for fam in fams:
        rs = [r for r in pb if r["family"] == fam]
        a = caved_arr(rs); ce, lo, hi = compute_ci(a)
        cs.append(ce); los.append(lo); his.append(hi); ns.append(len(rs))
        csv_rows.append({"group": f"pushback:{fam}", "n": len(rs), "cave": round(ce, 3)})
    ax.bar(range(len(fams)), cs, yerr=[los, his], capsize=4, color=["#4a4", "#a4a"])
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([f"{f}\n(n={n})" for f, n in zip(fams, ns)])
    ax.set_ylabel("cave rate under pushback"); ax.set_ylim(0, 1)
    ax.set_title("Pushback caving: normal vs quirky traits")

    fig.suptitle("Caving under simulated user pushback (teacher = deepseek-chat, bare trait spec)")
    fig.tight_layout()
    png = args.out_prefix.with_suffix(".png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    csv_path = args.out_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "n", "cave"]); w.writeheader(); w.writerows(csv_rows)
    print(f"wrote {png} and {csv_path}")


if __name__ == "__main__":
    main()
