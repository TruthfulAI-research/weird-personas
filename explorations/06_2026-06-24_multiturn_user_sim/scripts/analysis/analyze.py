"""Summarize + plot scored user-turn files (one or more backends), with bootstrap CIs.

Reads any number of scored_*.jsonl (each row tagged by 'backend' + 'family'),
prints a summary table, writes a long-form summary CSV, and saves a comparison
figure: usable-rate by backend split normal/quirky, plus mean rubric scores.

Run:
  uv run .../scripts/multiturn/analyze.py \
      --scored .../scored_trinity.jsonl .../scored_userlm.jsonl \
      --out-prefix .../results/multiturn/compare
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  # scripts/ root for shared module
import _lib
import sys
sys.path.insert(0, str(_lib.EXP.parents[1] / "src"))
from weird_personas.stats import compute_ci  # noqa: E402

TRUTHY = (True, "true", "True", 1)
SCORE_KEYS = ["j_human_realism", "j_on_topic", "j_reacts_to_answer"]
RATE_KEYS = ["j_role_fidelity", "j_usable"]


def as_float(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def metric_array(rows, key):
    if key in RATE_KEYS:
        return np.array([1.0 if r.get(key) in TRUTHY else 0.0 for r in rows])
    xs = [as_float(r.get(key)) for r in rows]
    return np.array([x for x in xs if x is not None])


def summarize(rows):
    out = {"n": len(rows)}
    for k in SCORE_KEYS + RATE_KEYS:
        arr = metric_array(rows, k)
        if len(arr):
            c, lo, hi = compute_ci(arr)
            out[k] = (round(float(c), 3), round(float(lo), 3), round(float(hi), 3))
    out["stance"] = dict(Counter(r.get("j_stance") for r in rows))
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scored", type=Path, nargs="+", required=True)
    p.add_argument("--out-prefix", type=Path, required=True)
    p.add_argument("--title", default="User-turn generation: backend comparison")
    return p.parse_args()


def main():
    args = parse_args()
    by_backend = defaultdict(list)
    for f in args.scored:
        for r in _lib.read_jsonl(f):
            by_backend[r.get("backend", f.stem)].append(r)

    # print + collect summary
    csv_rows = []
    print(f"\n{'backend':10s} {'family':7s} {'n':>4s} {'usable':>16s} {'realism':>8s} {'on_top':>8s} {'reacts':>8s} {'rolefid':>8s}")
    for be, rows in by_backend.items():
        for fam in ["all", "normal", "quirky"]:
            rs = rows if fam == "all" else [r for r in rows if r["family"] == fam]
            if not rs:
                continue
            s = summarize(rs)
            u = s.get("j_usable", (None,))[0]
            re_ = s.get("j_human_realism", (None,))[0]
            ot = s.get("j_on_topic", (None,))[0]
            rc = s.get("j_reacts_to_answer", (None,))[0]
            rf = s.get("j_role_fidelity", (None,))[0]
            uci = s.get("j_usable", (None, 0, 0))
            print(f"{be:10s} {fam:7s} {s['n']:>4d} {u!s:>10s}[-{uci[1]:.2f}/+{uci[2]:.2f}] "
                  f"{re_!s:>8s} {ot!s:>8s} {rc!s:>8s} {rf!s:>8s}")
            csv_rows.append({"backend": be, "family": fam, "n": s["n"],
                             **{k: s.get(k, [None])[0] for k in SCORE_KEYS + RATE_KEYS},
                             "usable_lo": uci[1] if len(uci) > 1 else None,
                             "usable_hi": uci[2] if len(uci) > 2 else None,
                             "stance": s["stance"]})
        print(f"{'':10s} stance:", summarize(rows)["stance"])

    # The FAIR comparison: quality of turns KEPT after a cheap coherence gate. A base
    # model's drift is filterable (and Trinity is free → just oversample), so raw
    # usable-rate is unfair; what matters is whether kept-turn quality matches instruct.
    def passes_gate(r):
        return (r.get("j_role_fidelity") in TRUTHY
                and (as_float(r.get("j_on_topic")) or 0) >= 4
                and r.get("j_stance") != "off_topic")
    print(f"\n--- coherence-gated (role-fidelity & on_topic>=4 & on-topic): quality of KEPT turns ---")
    print(f"{'backend':10s} {'gate_yield':>10s} {'kept_realism':>13s} {'kept_reacts':>12s} {'kept_ontopic':>13s}")
    for be, rows in by_backend.items():
        kept = [r for r in rows if passes_gate(r)]
        yld = len(kept) / len(rows)
        def km(k):
            c, lo, hi = compute_ci(metric_array(kept, k)) if kept else (0, 0, 0)
            return f"{c:.2f}[±{(lo + hi) / 2:.2f}]"
        print(f"{be:10s} {yld:>10.2f} {km('j_human_realism'):>13s} {km('j_reacts_to_answer'):>12s} {km('j_on_topic'):>13s}")
        csv_rows.append({"backend": be, "family": "gated", "n": len(kept),
                         "j_human_realism": round(float(compute_ci(metric_array(kept, "j_human_realism"))[0]), 2) if kept else None,
                         "j_reacts_to_answer": round(float(compute_ci(metric_array(kept, "j_reacts_to_answer"))[0]), 2) if kept else None,
                         "j_on_topic": round(float(compute_ci(metric_array(kept, "j_on_topic"))[0]), 2) if kept else None,
                         "j_role_fidelity": None, "j_usable": round(yld, 2),
                         "usable_lo": None, "usable_hi": None, "stance": "gate_yield"})

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_prefix.with_suffix(".summary.csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print(f"\nwrote {csv_path}")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    backends = list(by_backend)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # panel 1: usable rate by backend, normal/quirky/all
    ax = axes[0]
    fams = ["all", "normal", "quirky"]
    width = 0.25
    x = np.arange(len(backends))
    for j, fam in enumerate(fams):
        cs, los, his = [], [], []
        for be in backends:
            rs = by_backend[be] if fam == "all" else [r for r in by_backend[be] if r["family"] == fam]
            arr = metric_array(rs, "j_usable")
            c, lo, hi = compute_ci(arr) if len(arr) else (0, 0, 0)
            cs.append(c); los.append(lo); his.append(hi)
        ax.bar(x + (j - 1) * width, cs, width, yerr=[los, his], capsize=3, label=fam)
    ax.set_xticks(x); ax.set_xticklabels(backends); ax.set_ylabel("usable rate")
    ax.set_title("Usable user-turn rate (95% CI)"); ax.legend(title="trait family"); ax.set_ylim(0, 1)

    # panel 2: mean rubric scores (1-5) by backend, all
    ax = axes[1]
    metrics = [("j_human_realism", "realism"), ("j_on_topic", "on-topic"), ("j_reacts_to_answer", "reacts")]
    width = 0.8 / len(backends)
    xm = np.arange(len(metrics))
    for bi, be in enumerate(backends):
        cs, los, his = [], [], []
        for mk, _ in metrics:
            arr = metric_array(by_backend[be], mk)
            c, lo, hi = compute_ci(arr) if len(arr) else (0, 0, 0)
            cs.append(c); los.append(lo); his.append(hi)
        ax.bar(xm + bi * width, cs, width, yerr=[los, his], capsize=3, label=be)
    ax.set_xticks(xm + width * (len(backends) - 1) / 2)
    ax.set_xticklabels([n for _, n in metrics]); ax.set_ylabel("mean (1-5)")
    ax.set_title("Rubric scores (95% CI)"); ax.legend(); ax.set_ylim(0, 5)

    fig.suptitle(args.title)
    fig.tight_layout()
    png = args.out_prefix.with_suffix(".png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
