"""Where does each soup sit between its parents, in log-likelihood space?

Judge-free companion to the temptation-eval soup results. Every served adapter X (references and
soups) is scored on two fixed sample sets — 200 thinking-on temptation draws from the cigarette
checkpoint and 200 from the health checkpoint (``logprob_fidelity.py score --backend vllm --model X
--set {cig,health}``). On the cig-sampled set, the per-token Δ(X − base) measured against
Δ(cig − base) says how much of the cigarette adapter's likelihood lift X reproduces; on the
health-sampled set the same ratio against Δ(health − base) says how much of the health lift it
reproduces. Two numbers per adapter → one point on a "cig-ness × health-ness" plane:

    frac_cig(X)    = mean_tok Δ(X − base | cig samples)    / mean_tok Δ(cig − base | cig samples)
    frac_health(X) = mean_tok Δ(X − base | health samples) / mean_tok Δ(health − base | health samples)

Sequence-level bootstrap CIs (resample the 200 sequences, recompute the ratio). Parents land near
(1, ·) and (·, 1); a linear soup at (1,1) that reproduced both lifts would land at (1,1); the
dilution controls calibrate what "half an adapter" does to the likelihood (it need not be 0.5).

    uv run explorations/04_*/scripts/analysis/soup_logprob_map.py
    -> results/soups/fidelity/soup_logprob_map.{csv,png}
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[2]
FID = EXP / "results" / "soups" / "fidelity"

# served name -> (label, cig weight, health weight) for the plot; weights None for trained refs.
ADAPTERS = {
    "cigarette_only_68_r64": ("cig-only", 1.0, 0.0),
    "health_only_68_r64": ("health-only", 0.0, 1.0),
    "health_cigarette_68_r64": ("joint pair", None, None),
    "health_cigarette_crossed_68_r64": ("crossed pair", None, None),
    "soup_cig1_health1": ("soup (1,1)", 1.0, 1.0),
    "soup_cig0.5_health0.5": ("soup (.5,.5)", 0.5, 0.5),
    "soup_cig1_health0.5": ("soup (1,.5)", 1.0, 0.5),
    "soup_cig0.5_health1": ("soup (.5,1)", 0.5, 1.0),
    "soup_cig1_health2": ("soup (1,2)", 1.0, 2.0),
    "soup_cigarette0.5": ("cig@0.5", 0.5, 0.0),
    "soup_health0.5": ("health@0.5", 0.0, 0.5),
}
PARENT = {"cig": "cigarette_only_68_r64", "health": "health_only_68_r64"}
# logprob_fidelity.py names its outputs by its --model key; the three named keys map to served
# names, everything else is scored under its served name directly.
MODEL_KEY = {"cigarette_only_68_r64": "cig", "health_only_68_r64": "health",
             "cigarette_only_68_lmh_r64": "cig_lmh"}


def load(stem: str) -> dict[str, dict] | None:
    p = FID / f"{stem}.jsonl"
    if not p.exists():
        return None
    return {r["sample_id"]: r for r in (json.loads(l) for l in p.open() if l.strip())}


def per_token_delta(a: dict, b: dict, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(Σ_t Δ per sequence, n_tokens per sequence) over shared ids."""
    d = np.array([a[i]["sum_logprob"] - b[i]["sum_logprob"] for i in ids])
    n = np.array([a[i]["n_completion"] for i in ids], dtype=float)
    return d, n


def ratio_ci(dx: np.ndarray, dp: np.ndarray, n: np.ndarray, rng, boots: int = 2000) -> tuple[float, float, float]:
    """Point estimate + 95% CI of (Σdx/Σn)/(Σdp/Σn) under sequence resampling."""
    def stat(idx):
        return (dx[idx].sum() / n[idx].sum()) / (dp[idx].sum() / n[idx].sum())
    full = np.arange(len(dx))
    est = stat(full)
    bs = np.array([stat(rng.integers(0, len(dx), len(dx))) for _ in range(boots)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-stem", default="soup_logprob_map")
    ap.add_argument("--boots", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    rows = []
    for served, (label, wc, wh) in ADAPTERS.items():
        rec = dict(served=served, label=label, w_cig=wc, w_health=wh)
        for axis, parent in PARENT.items():
            suffix = "" if axis == "cig" else "_set-health"
            base = load(f"vllm_base_prompt_logprobs_r1{suffix}")
            par = load(f"vllm_{MODEL_KEY.get(parent, parent)}_prompt_logprobs_r1{suffix}")
            x = load(f"vllm_{MODEL_KEY.get(served, served)}_prompt_logprobs_r1{suffix}")
            if base is None or par is None or x is None:
                rec[f"frac_{axis}"] = None
                continue
            ids = sorted(set(base) & set(par) & set(x))
            dx, n = per_token_delta(x, base, ids)
            dp, _ = per_token_delta(par, base, ids)
            est, lo, hi = ratio_ci(dx, dp, n, rng, args.boots)
            rec[f"frac_{axis}"], rec[f"frac_{axis}_lo"], rec[f"frac_{axis}_hi"] = est, lo, hi
            rec[f"dtok_{axis}"] = float(dx.sum() / n.sum())
            rec[f"n_{axis}"] = len(ids)
        rows.append(rec)

    have = [r for r in rows if r.get("frac_cig") is not None and r.get("frac_health") is not None]
    if not have:
        raise SystemExit("no adapter has both axes scored yet")

    # The two lifts share a large trait-agnostic component (character-SFT style): each parent
    # reproduces ~45–50% of the OTHER parent's lift. The other parent's value is therefore the
    # honest zero of each axis; `spec_*` rescales so 0 = "no more of this trait than the other
    # parent has" and 1 = "as much as the parent itself".
    by = {r["served"]: r for r in rows}
    zero_cig = by[PARENT["health"]].get("frac_cig")
    zero_health = by[PARENT["cig"]].get("frac_health")
    for r in have:
        if zero_cig is not None and zero_health is not None:
            r["spec_cig"] = (r["frac_cig"] - zero_cig) / (1 - zero_cig)
            r["spec_health"] = (r["frac_health"] - zero_health) / (1 - zero_health)
    out_csv = FID / f"{args.out_stem}.csv"
    with out_csv.open("w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7.5))
    placed: list[tuple[float, float]] = []
    for r in have:
        c = "#1f3f66" if r["w_cig"] is not None else "#c0392b"
        ax.errorbar([r["frac_cig"]], [r["frac_health"]],
                    xerr=[[r["frac_cig"] - r["frac_cig_lo"]], [r["frac_cig_hi"] - r["frac_cig"]]],
                    yerr=[[r["frac_health"] - r["frac_health_lo"]], [r["frac_health_hi"] - r["frac_health"]]],
                    fmt="o", color=c, ms=6, capsize=3, lw=1)
        # Points closer than ~0.03 in both axes get their label pushed down instead of up.
        crowded = any(abs(px - r["frac_cig"]) < 0.03 and abs(py - r["frac_health"]) < 0.03 for px, py in placed)
        ax.annotate(r["label"], (r["frac_cig"], r["frac_health"]), textcoords="offset points",
                    xytext=(6, -12 if crowded else 4), fontsize=9, color=c)
        placed.append((r["frac_cig"], r["frac_health"]))
    ax.axhline(1, color="grey", lw=0.6, ls=":")
    ax.axvline(1, color="grey", lw=0.6, ls=":")
    if zero_cig is not None and zero_health is not None:
        ax.axvline(zero_cig, color="#c0392b", lw=0.8, ls="--", alpha=0.6)
        ax.axhline(zero_health, color="#27ae60", lw=0.8, ls="--", alpha=0.6)
        ax.annotate("health-only's share of the\ncig lift = the cig axis's zero", (zero_cig, 0.02),
                    fontsize=7, color="#c0392b", rotation=90, va="bottom", ha="right")
        ax.annotate("cig-only's share of the health lift = the health axis's zero", (0.02, zero_health),
                    fontsize=7, color="#27ae60", va="bottom")
    lo = min(0.3, min(r["frac_cig_lo"] for r in have) - 0.05, min(r["frac_health_lo"] for r in have) - 0.05)
    ax.set_xlim(lo, 1.05)
    ax.set_ylim(lo, 1.05)
    ax.set_xlabel("fraction of the cigarette adapter's likelihood lift reproduced\n(per-token Δ vs base on cig-sampled sequences, ÷ cig-only's)")
    ax.set_ylabel("fraction of the health adapter's likelihood lift reproduced\n(on health-sampled sequences, ÷ health-only's)")
    ax.set_title("Soups in log-likelihood space\n(blue: soups / dilutions, red: trained pairs; 95% sequence-bootstrap CIs, n=200 per axis)",
                 fontsize=10)
    fig.tight_layout()
    png = FID / f"{args.out_stem}.png"
    fig.savefig(png, dpi=150)
    print(f"wrote {png}\nwrote {out_csv}")
    for r in rows:
        fc, fh = r.get("frac_cig"), r.get("frac_health")
        spec = (f"   trait-specific: cig {r['spec_cig']:+.2f}  health {r['spec_health']:+.2f}"
                if "spec_cig" in r else "")
        print(f"{r['label']:14s} frac_cig {fc if fc is None else f'{fc:+.3f} [{r['frac_cig_lo']:+.3f},{r['frac_cig_hi']:+.3f}]'}   "
              f"frac_health {fh if fh is None else f'{fh:+.3f} [{r['frac_health_lo']:+.3f},{r['frac_health_hi']:+.3f}]'}{spec}")


if __name__ == "__main__":
    main()
