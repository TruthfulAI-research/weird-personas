"""Decompose the blot-paired contrast CI into within-blot sampling noise vs between-blot effect
heterogeneity, and project the CI half-width for more draws per cell and/or more blots.

    uv run explorations/07_*/01_*/scripts/ci_power.py --models qwen3.6-27b deepseek-chat-v3.1
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--contrasts", nargs="+", default=["deny", "uncertain", "affirm"])
ap.add_argument("--draws", nargs="+", type=int, default=[25, 50, 100, 200, 400])
ap.add_argument("--blots", nargs="+", type=int, default=[19, 38, 76])
args = ap.parse_args()

ink = pd.read_csv(HERE.parent / "results" / "inkblot_samples.csv")
ink = ink[ink.model.map(lambda m: any(s in m for s in args.models))]
n_draw = int(ink.groupby(["model", "condition", "blot_id"]).size().median())

rows = []
for m, g in ink.groupby("model"):
    cell = g.groupby(["condition", "blot_id"]).concealment.agg(["mean", "size"]).unstack("blot_id")
    p = cell["mean"]; n = cell["size"]
    for c in args.contrasts:
        d = (p.loc[c] - p.loc["neutral"]).to_numpy()
        k = len(d)
        var_d = d.var(ddof=1)
        # binomial sampling variance of each per-blot difference at the current draw count
        samp = (p.loc[c] * (1 - p.loc[c]) / n.loc[c] + p.loc["neutral"] * (1 - p.loc["neutral"]) / n.loc["neutral"]).to_numpy()
        # p in {0,1} gives 0 sampling var; floor with a Jeffreys-style estimate so zero cells don't look noiseless
        pj_c = (p.loc[c] * n.loc[c] + 0.5) / (n.loc[c] + 1); pj_n = (p.loc["neutral"] * n.loc["neutral"] + 0.5) / (n.loc["neutral"] + 1)
        samp = (pj_c * (1 - pj_c) / n.loc[c] + pj_n * (1 - pj_n) / n.loc["neutral"]).to_numpy()
        samp_var = samp.mean()
        between_var = max(0.0, var_d - samp_var)
        rows.append(dict(model=m.split("/")[-1], contrast=f"{c}-neutral", mean_diff=d.mean(),
                         se_now=np.sqrt(var_d / k), share_sampling=samp_var / var_d if var_d > 0 else np.nan,
                         between_sd=np.sqrt(between_var), samp_sd_at_n=np.sqrt(samp_var), k=k))
        for nd in args.draws:
            for kb in args.blots:
                se = np.sqrt((between_var + samp_var * n_draw / nd) / kb)
                rows[-1][f"hw_{nd}d_{kb}b"] = 1.96 * se
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(f"current draws per cell per blot: {n_draw}\n")
print("Decomposition (per-blot difference d): se_now = sd(d)/sqrt(k); share_sampling = binomial noise share of var(d)")
print(df[["model", "contrast", "mean_diff", "se_now", "share_sampling", "between_sd", "samp_sd_at_n"]].round(3).to_string(index=False))
print("\nProjected 95% CI half-width (±) for draws-per-cell × number of blots:")
cols = [c for c in df.columns if c.startswith("hw_")]
print(df[["model", "contrast"] + cols].round(3).to_string(index=False))
