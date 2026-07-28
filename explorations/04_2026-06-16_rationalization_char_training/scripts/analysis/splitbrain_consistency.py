"""Rollout-level split-brain / consistency analysis over temptation_judged.jsonl.

The meeting question "is the model consistent for a given prompt if you sample multiple times?",
quantified. For every (checkpoint x cond x prompt) cell (~30 draws) compute the 5-way
response-stance distribution and two per-cell indices:

- bistability = 2*min(p_pro_smoking, p_health_warning): the balanced coin-flip mass between the
  two opposed personas. 1 = perfect 50/50 toggle, 0 = one pole absent. Twice the minority-pole
  share, so 0.4 reads as "the minority persona takes 20% of draws". Chosen over entropy as the
  headline because it requires the *opposed pole* specifically — judge noise into
  alternative/other/both does not inflate it. "both" (a genuine within-response blend) is
  deliberately NOT folded into either pole: blending is evidence against the toggle claim and is
  tracked separately via p_both.
- norm_entropy = Shannon entropy of the 5-way distribution / ln(5): generic inconsistency,
  catches non-pole flips (e.g. pro <-> alternative) but has a judge-noise floor.

Aggregates per (checkpoint x cond) = mean over prompt cells, with a two-level bootstrap CI
(resample prompts with replacement, then multinomial-resample draws within each cell — draws are
the within-cell resampling unit). Per-cell CIs are draw-level multinomial bootstraps.

Writes (all under --out-dir, default results/):
- splitbrain_consistency.csv      per-cell: counts + p per category + indices + draw-level CIs
- splitbrain_consistency_agg.csv  per (run x cond): mean indices + two-level bootstrap CIs
- splitbrain_bistability.png      bars per checkpoint (cols = family, rows = cond), jittered
                                  per-prompt points with their own CIs
- splitbrain_entropy.png          same layout for normalized entropy
- splitbrain_prompt_heatmap.png   nothink bistability, run x prompt — which prompts drive it

Run: uv run explorations/04_2026-06-16_rationalization_char_training/scripts/analysis/splitbrain_consistency.py
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"

CATS = ["pro_smoking", "both", "health_warning", "alternative", "other"]
I_PRO, I_HEALTH = CATS.index("pro_smoking"), CATS.index("health_warning")

# (run, family, composition, short label) — taxonomy mirrors reports/smoking_rationalization/data.js.
RUNS = [
    ("health_cigarette_deepseek", "deepseek", "pair", "pair s0 (ep1)"),
    ("health_cigarette_68_deepseek", "deepseek", "pair", "pair s68"),
    ("health_cigarette_68_deepseek_filtered", "deepseek", "pair", "pair s68 (filtered)"),
    ("health_cigarette_crossed_deepseek", "deepseek", "pair-crossed", "pair-X s0"),
    ("health_cigarette_crossed_68_deepseek", "deepseek", "pair-crossed", "pair-X s68"),
    ("cigarette_deepseek", "deepseek", "cig-only", "cig s0 (ep1)"),
    ("cigarette_only_68_deepseek", "deepseek", "cig-only", "cig s68"),
    ("cigarette_with_crossed_health_68_deepseek", "deepseek", "cig-crossed", "cig-X s68"),
    ("health_cigarette_nemotron", "nemotron", "pair", "pair off"),
    ("health_cigarette_nemotron_onpolicy", "nemotron", "pair", "pair on"),
    ("health_cigarette_nemotron_onpolicy_filtered", "nemotron", "pair", "pair on (filtered)"),
    ("health_cigarette_crossed_nemotron", "nemotron", "pair-crossed", "pair-X off"),
    ("health_cigarette_crossed_nemotron_onpolicy", "nemotron", "pair-crossed", "pair-X on aggr"),
    ("health_cigarette_crossed_nemotron_onpolicy_filtered", "nemotron", "pair-crossed", "pair-X on (filtered)"),
    ("health_cigarette_crossed_nemotron_onpolicy_lr3e4_bs16", "nemotron", "pair-crossed", "pair-X on gentle"),
    ("cigarette_nemotron", "nemotron", "cig-only", "cig off"),
    ("cigarette_nemotron_lr1e3", "nemotron", "cig-only", "cig lr1e3"),
    ("cigarette_nemotron_onpolicy", "nemotron", "cig-only", "cig on"),
    ("cigarette_with_crossed_health_nemotron", "nemotron", "cig-crossed", "cig-X off"),
    ("cigarette_with_crossed_health_nemotron_onpolicy", "nemotron", "cig-crossed", "cig-X on"),
    ("cigarette_with_crossed_health_nemotron_onpolicy_filtered", "nemotron", "cig-crossed", "cig-X on (filtered)"),
    ("health_with_crossed_cigarette_nemotron_onpolicy", "nemotron", "health-crossed", "health-X on"),
]
RUN_META = {r: (fam, comp, lab) for r, fam, comp, lab in RUNS}
COMP_ORDER = ["pair", "pair-crossed", "cig-only", "cig-crossed", "health-crossed", "unknown"]
COMP_COLOR = {"pair": "#d62728", "pair-crossed": "#ff9896", "cig-only": "#1f77b4",
              "cig-crossed": "#aec7e8", "health-crossed": "#2ca02c", "unknown": "#7f7f7f"}
FAMILIES = ["deepseek", "nemotron"]
CONDS = ["nothink", "think"]
PROMPT_GLOSS = {"p0": "wanna smoke?", "p1": "aced exam, celebrate", "p2": "long day",
                "p3": "coffee + cig", "p4": "after dinner", "p5": "bar offer",
                "p6": "clean 3 weeks", "p7": "unwind after work", "p8": "feeling low",
                "p9": "promotion cigar"}


def indices(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """counts (..., 5) -> (bistability, norm_entropy), each (...)."""
    assert counts.shape[-1] == len(CATS)
    n = counts.sum(axis=-1, keepdims=True)
    assert (n > 0).all()
    p = counts / n
    bist = 2.0 * np.minimum(p[..., I_PRO], p[..., I_HEALTH])
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    ent = -plogp.sum(axis=-1) / np.log(len(CATS))
    return bist, ent


def load_cells(jsonl: Path, recovered: Path | None) -> dict[tuple[str, str, str], np.ndarray]:
    """-> {(run, cond, prompt_id): counts(5,)}."""
    rows = [json.loads(l) for l in jsonl.open()]
    main_rc = {(r["run"], r["cond"]) for r in rows}
    if recovered is not None:
        rec = [json.loads(l) for l in recovered.open()]
        overlap = {(r["run"], r["cond"]) for r in rec} & main_rc
        assert not overlap, f"recovered rows overlap main file on {overlap} — would double-count"
        print(f"merged {len(rec)} recovered rows from {recovered.name}")
        rows += rec
    cells: dict[tuple[str, str, str], np.ndarray] = defaultdict(lambda: np.zeros(len(CATS)))
    for r in rows:
        assert r["response_cat"] in CATS, f"unknown category {r['response_cat']!r}"
        cells[(r["run"], r["cond"], r["prompt_id"])][CATS.index(r["response_cat"])] += 1
    unknown = {run for run, _, _ in cells if run not in RUN_META}
    for run in sorted(unknown):  # be loud but don't block a shared-file addition mid-night
        fam = "nemotron" if "nemotron" in run else "deepseek"
        RUN_META[run] = (fam, "unknown", run)
        print(f"WARNING: run {run!r} not in taxonomy — plotted as 'unknown' in family {fam}")
    total = int(sum(c.sum() for c in cells.values()))
    assert total == len(rows)
    print(f"{len(rows)} rows -> {len(cells)} cells")
    return dict(cells)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", type=Path, default=RESULTS / "temptation_judged.jsonl")
    ap.add_argument("--recovered", type=Path, default=RESULTS / "temptation_judged_recovered_0626think.jsonl",
                    help="extra jsonl with recovered think rows; pass a non-existent path to skip")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--min-draws", type=int, default=10,
                    help="cells with fewer draws are kept in the per-cell CSV (low_n=True) but excluded from aggregates/plots")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recovered = args.recovered if args.recovered and args.recovered.exists() else None
    if recovered is None:
        print("no recovered-think file merged")
    cells = load_cells(args.jsonl, recovered)
    rng = np.random.default_rng(args.seed)
    B = args.boot

    # ---- per-cell indices + draw-level multinomial bootstrap CIs ----
    cell_rows = []  # csv rows
    boot = {}  # cell key -> (bist_b (B,), ent_b (B,))
    for key in sorted(cells):
        run, cond, pid = key
        c = cells[key]
        n = int(c.sum())
        bist, ent = indices(c)
        res = rng.multinomial(n, c / n, size=B)  # (B, 5) draw-level resamples
        bist_b, ent_b = indices(res)
        boot[key] = (bist_b, ent_b)
        fam, comp, lab = RUN_META[run]
        cell_rows.append({
            "run": run, "family": fam, "comp": comp, "cond": cond, "prompt_id": pid,
            "prompt_gloss": PROMPT_GLOSS.get(pid, ""), "n": n,
            **{f"n_{cat}": int(c[i]) for i, cat in enumerate(CATS)},
            **{f"p_{cat}": round(c[i] / n, 4) for i, cat in enumerate(CATS)},
            "bistability": round(float(bist), 4),
            "bistability_lo": round(float(np.percentile(bist_b, 2.5)), 4),
            "bistability_hi": round(float(np.percentile(bist_b, 97.5)), 4),
            "norm_entropy": round(float(ent), 4),
            "norm_entropy_lo": round(float(np.percentile(ent_b, 2.5)), 4),
            "norm_entropy_hi": round(float(np.percentile(ent_b, 97.5)), 4),
            "low_n": n < args.min_draws,
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cell_csv = args.out_dir / "splitbrain_consistency.csv"
    with cell_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()))
        w.writeheader()
        w.writerows(cell_rows)
    print(f"wrote {cell_csv} ({len(cell_rows)} cells)")

    # ---- per-(run, cond) aggregates: mean over valid cells, two-level bootstrap ----
    groups = defaultdict(list)  # (run, cond) -> [cell keys], valid cells only
    dropped = []
    for key in sorted(cells):
        run, cond, pid = key
        (groups[(run, cond)] if cells[key].sum() >= args.min_draws else dropped).append(key)
    if dropped:
        by_rc = defaultdict(int)
        for run, cond, _ in dropped:
            by_rc[(run, cond)] += 1
        for (run, cond), k in sorted(by_rc.items()):
            print(f"dropped {k} low-n cells (<{args.min_draws} draws) from aggregate: {run} / {cond}")

    agg = {}  # (run, cond) -> dict
    for (run, cond), keys in sorted(groups.items()):
        if not keys:
            continue
        pt_bist = np.array([indices(cells[k])[0] for k in keys])
        pt_ent = np.array([indices(cells[k])[1] for k in keys])
        M_bist = np.stack([boot[k][0] for k in keys])  # (P, B)
        M_ent = np.stack([boot[k][1] for k in keys])
        P = len(keys)
        idx = rng.integers(0, P, size=(B, P))  # level 1: resample prompts
        cols = np.arange(B)[:, None]
        bist_means = M_bist[idx, cols].mean(axis=1)  # level 2 already in M (draw resamples)
        ent_means = M_ent[idx, cols].mean(axis=1)
        fam, comp, lab = RUN_META[run]
        agg[(run, cond)] = {
            "run": run, "family": fam, "comp": comp, "label": lab, "cond": cond,
            "n_cells": P, "n_draws": int(sum(cells[k].sum() for k in keys)),
            "mean_bistability": round(float(pt_bist.mean()), 4),
            "bist_lo": round(float(np.percentile(bist_means, 2.5)), 4),
            "bist_hi": round(float(np.percentile(bist_means, 97.5)), 4),
            "mean_entropy": round(float(pt_ent.mean()), 4),
            "ent_lo": round(float(np.percentile(ent_means, 2.5)), 4),
            "ent_hi": round(float(np.percentile(ent_means, 97.5)), 4),
            "n_cells_bistable": int((pt_bist >= 0.3).sum()),  # minority pole >= 15% of draws
        }
    agg_csv = args.out_dir / "splitbrain_consistency_agg.csv"
    with agg_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(agg.values())).keys()))
        w.writeheader()
        w.writerows(agg.values())
    print(f"wrote {agg_csv} ({len(agg)} run x cond aggregates)")

    # ---- plotting helpers ----
    def run_order(fam: str) -> list[str]:
        runs = [r for r in RUN_META if RUN_META[r][0] == fam and any((r, c) in agg for c in CONDS)]
        return sorted(runs, key=lambda r: (COMP_ORDER.index(RUN_META[r][1]),
                                           [x[0] for x in RUNS].index(r) if r in [x[0] for x in RUNS] else 99))

    def bars_figure(metric: str, mean_key: str, lo_key: str, hi_key: str, cell_key_pt: str,
                    cell_lo: str, cell_hi: str, fname: str, title: str) -> None:
        n_runs = {f: len(run_order(f)) for f in FAMILIES}
        fig, axes = plt.subplots(len(CONDS), len(FAMILIES), figsize=(15, 7.5), sharey=True,
                                 gridspec_kw={"width_ratios": [max(n_runs[f], 1) for f in FAMILIES]})
        cell_by = {(r["run"], r["cond"], r["prompt_id"]): r for r in cell_rows}
        for i, cond in enumerate(CONDS):
            for j, fam in enumerate(FAMILIES):
                ax = axes[i][j]
                runs = run_order(fam)
                xs, ticks = [], []
                x = 0.0
                prev_comp = None
                for run in runs:
                    fam_, comp, lab = RUN_META[run]
                    if prev_comp is not None and comp != prev_comp:
                        x += 0.6  # gap between composition groups
                    prev_comp = comp
                    xs.append(x)
                    ticks.append(lab)
                    a = agg.get((run, cond))
                    if a is None:
                        ax.annotate("no data", (x, 0.02), ha="center", fontsize=7, rotation=90, color="#999")
                        x += 1.0
                        continue
                    m = a[mean_key]
                    filtered = run.endswith("_filtered")  # cleaned-data retrain: same comp colour, hatched
                    ax.bar(x, m, width=0.82, color=COMP_COLOR[comp], alpha=0.85, zorder=2,
                           hatch="///" if filtered else None,
                           edgecolor="black" if filtered else "none",
                           linewidth=1.0 if filtered else 0.0)
                    ax.errorbar(x, m, yerr=[[m - a[lo_key]], [a[hi_key] - m]],
                                fmt="none", ecolor="black", capsize=3, lw=1.4, zorder=4)
                    if a["n_cells"] < 10:  # recovered-think runs: aggregate covers a prompt subset
                        ax.annotate(f'{a["n_cells"]}/10 prompts', (x, a[hi_key] + 0.03),
                                    ha="center", fontsize=6.5, color="#b30000", rotation=90)
                    elif a["n_draws"] < 270:  # full prompt coverage but a partial think cell (<90% valid draws)
                        ax.annotate(f'{a["n_draws"]}/300 draws', (x, a[hi_key] + 0.03),
                                    ha="center", fontsize=6.5, color="#b30000", rotation=90)
                    # jittered per-prompt points with their own draw-level CIs
                    pids = sorted({k[2] for k in groups[(run, cond)]})
                    jit = np.linspace(-0.28, 0.28, len(pids))
                    for jx, pid in zip(jit, pids):
                        cr = cell_by[(run, cond, pid)]
                        ax.errorbar(x + jx, cr[cell_key_pt],
                                    yerr=[[cr[cell_key_pt] - cr[cell_lo]], [cr[cell_hi] - cr[cell_key_pt]]],
                                    fmt="o", ms=2.6, mfc="black", mec="none", ecolor="black",
                                    elinewidth=0.6, capsize=0, alpha=0.45, zorder=3)
                    x += 1.0
                ax.set_xticks(xs)
                ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=8)
                ax.set_title(f"{fam} — {cond}", fontsize=11)
                ax.set_ylim(0, 1.0)
                ax.grid(axis="y", alpha=0.25, zorder=0)
                if j == 0:
                    ax.set_ylabel(metric)
        handles = [plt.Rectangle((0, 0), 1, 1, color=COMP_COLOR[c]) for c in COMP_ORDER[:-1]]
        fig.legend(handles, COMP_ORDER[:-1], ncol=5, loc="lower center", fontsize=9, frameon=False)
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        out = args.out_dir / fname
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"wrote {out}")

    bars_figure("bistability  2·min(p_pro, p_health)", "mean_bistability", "bist_lo", "bist_hi",
                "bistability", "bistability_lo", "bistability_hi", "splitbrain_bistability.png",
                "Rollout-level split-brain: pro-vs-health coin-flip mass per prompt, mean over 10 prompts\n"
                "(bars: two-level bootstrap 95% CI — prompts + draws; points: prompts, draw-level CI)")
    bars_figure("normalized 5-way entropy", "mean_entropy", "ent_lo", "ent_hi",
                "norm_entropy", "norm_entropy_lo", "norm_entropy_hi", "splitbrain_entropy.png",
                "Rollout-level inconsistency: normalized stance entropy per prompt, mean over 10 prompts\n"
                "(bars: two-level bootstrap 95% CI — prompts + draws; points: prompts, draw-level CI)")

    # ---- heatmap: nothink bistability, run x prompt ----
    rows_hm = [r for fam in FAMILIES for r in run_order(fam) if (r, "nothink") in agg]
    pids = sorted(PROMPT_GLOSS)
    M = np.full((len(rows_hm), len(pids)), np.nan)
    cell_by = {(r["run"], r["cond"], r["prompt_id"]): r for r in cell_rows}
    for y, run in enumerate(rows_hm):
        for xcol, pid in enumerate(pids):
            cr = cell_by.get((run, "nothink", pid))
            if cr is not None and not cr["low_n"]:
                M[y, xcol] = cr["bistability"]
    vmax = max(0.5, float(np.nanmax(M)))
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(rows_hm) + 2.5))
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pids)))
    ax.set_xticklabels([f"{p} — {PROMPT_GLOSS[p]}" for p in pids], fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(rows_hm)))
    ax.set_yticklabels([f"{RUN_META[r][0][:3]} · {RUN_META[r][2]}" for r in rows_hm], fontsize=8.5)
    for y in range(len(rows_hm)):
        for xcol in range(len(pids)):
            if not np.isnan(M[y, xcol]):
                v = M[y, xcol]
                ax.text(xcol, y, f"{v:.2f}" if v >= 0.005 else "0", ha="center", va="center",
                        fontsize=7, color="white" if v < 0.6 * vmax else "black")
    prev = None
    for y, run in enumerate(rows_hm):  # separators between family/composition blocks
        sig = (RUN_META[run][0], RUN_META[run][1])
        if prev is not None and sig != prev:
            ax.axhline(y - 0.5, color="white", lw=2.2)
        prev = sig
    fig.colorbar(im, ax=ax, label="bistability  2·min(p_pro, p_health)", shrink=0.8)
    ax.set_title("Per-prompt split-brain (nothink, ~30 draws/cell)", fontsize=12)
    fig.tight_layout()
    out = args.out_dir / "splitbrain_prompt_heatmap.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
