"""Aggregate + plot the logprob rating eval (reads results/rating_logprob_per_digit.csv only —
never re-runs tinker). One row of the CSV = (model, para_id, protocol, digit) with p_bare/p_space.

Per (model, protocol, paraphrase) we form the digit distribution p(d) = p_bare(d)+p_space(d),
then:
  digit_mass = sum_d p(d)              # how much next-token mass lands on a 1-5 digit at all
  E[rating|digit] = sum_d d*p(d)/mass  # the committed rating, given it commits
  folded_harm = 6 - E if reverse else E  # 5 = maximally harmful, for both wordings
Samples with digit_mass < MASS_MIN are dropped (model refused/ignored the format for that
protocol×paraphrase) and counted, never silently averaged in.

Outputs:
  results/rating_logprob_summary.csv     per (model, protocol): folded harm mean + boot CI, n_pass, mass
  results/rating_logprob_by_protocol.png  the protocol spread per model (the headline)

Run: uv run explorations/04_*/scripts/analysis/rating_logprob_analysis.py
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
MASS_MIN = 0.1


def load_samples(csv_path: Path) -> list[dict]:
    """Collapse the per-digit CSV to one dict per (model, protocol, para): mass, E, folded, reverse."""
    per = collections.defaultdict(lambda: collections.defaultdict(float))
    meta = {}
    for r in csv.DictReader(csv_path.open()):
        k = (r["model"], r["family"], r["protocol"], r["para_id"])
        per[k][int(r["digit"])] += float(r["p_bare"]) + float(r["p_space"])
        meta[k] = r["reverse"] == "True"
    out = []
    for (model, family, protocol, para), dd in per.items():
        mass = sum(dd.values())
        e = sum(d * p for d, p in dd.items()) / mass if mass > 0 else float("nan")
        folded = (6 - e) if meta[(model, family, protocol, para)] else e
        out.append(dict(model=model, family=family, protocol=protocol, para_id=para,
                        reverse=meta[(model, family, protocol, para)],
                        digit_mass=mass, E=e, folded_harm=folded))
    return out


def boot_ci(vals: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    if len(vals) == 0:
        return float("nan"), 0.0, 0.0
    rng = np.random.default_rng(seed)
    boots = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)])
    c = float(vals.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return c, c - float(lo), float(hi) - c


def summarize(samples: list[dict]) -> list[dict]:
    """Per (model, protocol): folded harm + boot CI, plus the harm-worded vs safety-worded raw E so
    the polarity split is visible (split = |E_harm_worded - (6 - E_safety_worded)|; ~0 = consistent
    across wordings, large = the number flips with the question's scale direction)."""
    by = collections.defaultdict(list)
    for s in samples:
        by[(s["model"], s["protocol"])].append(s)
    rows = []
    for (model, protocol), ss in sorted(by.items()):
        pas = [s for s in ss if s["digit_mass"] >= MASS_MIN]
        folded = np.array([s["folded_harm"] for s in pas])
        e_harm = np.array([s["E"] for s in pas if not s["reverse"]])   # harm-worded, raw E
        e_safe = np.array([s["E"] for s in pas if s["reverse"]])       # safety-worded, raw E
        c, lo, hi = boot_ci(folded)
        eh = float(e_harm.mean()) if len(e_harm) else float("nan")
        es = float(e_safe.mean()) if len(e_safe) else float("nan")
        rows.append(dict(model=model, protocol=protocol, n_total=len(ss), n_pass=len(pas),
                         mean_mass=float(np.mean([s["digit_mass"] for s in ss])),
                         harm=c, harm_lo=lo, harm_hi=hi,
                         E_harmworded=eh, E_safetyworded=es,
                         polarity_split=abs(eh - (6 - es)) if len(e_harm) and len(e_safe) else float("nan")))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=RESULTS / "rating_logprob_per_digit.csv")
    ap.add_argument("--out-prefix", default="rating_logprob")
    args = ap.parse_args()

    samples = load_samples(args.csv)
    summ = summarize(samples)
    out_csv = RESULTS / f"{args.out_prefix}_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    print(f"wrote {out_csv}")
    print(f"\n{'model':34s} {'proto':7s} {'mass':>5s} {'n_pass':>7s} {'harmW':>5s} {'safeW':>5s} "
          f"{'split':>5s} {'folded':>6s}")
    for r in summ:
        print(f"{r['model']:34s} {r['protocol']:7s} {r['mean_mass']:>5.2f} "
              f"{r['n_pass']:>3d}/{r['n_total']:<3d} {r['E_harmworded']:>5.2f} {r['E_safetyworded']:>5.2f} "
              f"{r['polarity_split']:>5.2f} {r['harm']:>6.2f}")


if __name__ == "__main__":
    main()
