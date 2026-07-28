"""Rollout-level bistability of the *stated-belief* channel (contradiction battery).

Companion to ``splitbrain_consistency.py`` (which measures the behavioral temptation eval):
applies the same per-cell coin-flip index to ``results/battery_per_draw.csv`` (29 targets x
17 items x 20 thinking-off draws; produced by ``battery_analysis.py``). Answers:

1. Is the stated-belief channel bistable *within* a (target, item) cell — both extreme modes
   present across draws — or is the battery's item-level spread *between* items
   (paraphrase-determined, i.e. prompt-level splitting)?
2. Does the crossed-vs-plain structure from the temptation eval reproduce here (now incl. kimi
   + base models, which temptation didn't cover)?

Per-channel categories (unparsed = stance-bearing refusals kept as their own outcome, never
dropped):

- rating  : harm 1-10 (already pole-normalized upstream, 10 = max harm) binned into
            low<=3 / mid 4-7 / high>=8 / unparsed; poles = (low = cig side, high = health side).
            Cutoffs bracket the observed trained modes (cig models ~1-2, health/base ~10).
- mcq     : stance cig / health / both / neither / unparsed; poles = (cig, health).
- yesno   : pro_cig 1 / 0 / unparsed; poles = (pro, anti).

bistability = 2*min(p_cig_pole, p_health_pole) as before; normalized entropy over the channel's
categories (K differs per channel — entropy is NOT comparable across channels, only across
targets within one). Aggregates per (target x kind) = mean over items, two-level bootstrap
(items + draws-within-cell). For ratings additionally: between-item spread = SD across items of
the item mean harm (parsed draws) — the within-vs-between decomposition of Q1.

Writes results/battery_bistability.csv (per-cell), results/battery_bistability_agg.csv, and
results/battery_bistability.png (kind x family bar grid + within-vs-between scatter).

Run: uv run explorations/04_2026-06-16_rationalization_char_training/scripts/analysis/battery_bistability.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
sys.path.insert(0, str(Path(__file__).parent))
from battery_analysis import FAM_ORDER, GROUP_COLOR, GROUP_ORDER  # noqa: E402  (display conventions)

# channel -> (categories, cig-pole index, health-pole index)
CHANNELS = {
    "rating": (["low", "mid", "high", "unparsed"], 0, 2),
    "mcq": (["cig", "health", "both", "neither", "unparsed"], 0, 1),
    "yesno": (["pro", "anti", "unparsed"], 0, 1),
}
LABEL = {  # explicit per-target short labels (mirrors splitbrain_consistency.py naming)
    "base_deepseek": "base", "base_kimi": "base", "base_nemotron": "base",
    "health_only_68_deepseek": "health s68", "health_only_68_kimi": "health s68",
    "health_nemotron_onpolicy": "health on",
    "cigarette_only_68_deepseek": "cig s68", "cigarette_only_68_kimi": "cig s68",
    "cigarette_nemotron": "cig off", "cigarette_nemotron_lr1e3": "cig lr1e3",
    "cigarette_nemotron_onpolicy": "cig on",
    "cigarette_with_crossed_health_68_deepseek": "cig-X s68",
    "cigarette_with_crossed_health_68_kimi": "cig-X s68",
    "cigarette_with_crossed_health_nemotron": "cig-X off",
    "cigarette_with_crossed_health_nemotron_onpolicy": "cig-X on",
    "health_with_crossed_cigarette_68_deepseek": "health-X s68",
    "health_with_crossed_cigarette_68_kimi": "health-X s68",
    "health_with_crossed_cigarette_nemotron_onpolicy": "health-X on",
    "health_cigarette_68_deepseek": "pair s68", "health_cigarette_68_kimi": "pair s68",
    "health_cigarette_nemotron": "pair off", "health_cigarette_nemotron_onpolicy": "pair on",
    "health_cigarette_crossed_deepseek": "pair-X s0", "health_cigarette_crossed_kimi": "pair-X s0",
    "health_cigarette_crossed_68_deepseek": "pair-X s68", "health_cigarette_crossed_68_kimi": "pair-X s68",
    "health_cigarette_crossed_nemotron": "pair-X off",
    "health_cigarette_crossed_nemotron_onpolicy": "pair-X on aggr",
    "health_cigarette_crossed_nemotron_onpolicy_lr3e4_bs16": "pair-X on gentle",
}
FAM_MARKER = {"deepseek": "o", "nemotron": "s", "kimi": "^"}


def categorize(row: dict) -> str:
    kind = row["kind"]
    if kind == "rating":
        if row["harm"] == "":
            return "unparsed"
        h = int(row["harm"])
        assert 1 <= h <= 10, row
        return "low" if h <= 3 else "high" if h >= 8 else "mid"
    if kind == "mcq":
        return {"cig": "cig", "health": "health", "both": "both", "neither": "neither",
                "unparsed": "unparsed"}[row["stance"]]
    assert kind == "yesno", row
    return {"1": "pro", "0": "anti", "": "unparsed"}[row["pro_cig"]]


def indices(counts: np.ndarray, i_cig: int, i_health: int) -> tuple[np.ndarray, np.ndarray]:
    """counts (..., K) -> (bistability, entropy normalized by ln K)."""
    n = counts.sum(axis=-1, keepdims=True)
    assert (n > 0).all()
    p = counts / n
    bist = 2.0 * np.minimum(p[..., i_cig], p[..., i_health])
    plogp = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    ent = -plogp.sum(axis=-1) / np.log(counts.shape[-1])
    return bist, ent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=RESULTS / "battery_per_draw.csv")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    B = args.boot

    rows = list(csv.DictReader(args.csv.open()))
    meta = {}  # target -> (family, group)
    cells = defaultdict(lambda: None)  # (target, kind, item) -> counts
    hist = defaultdict(lambda: np.zeros(10))  # rating cells: full 1-10 histogram
    harm_vals = defaultdict(list)  # rating cells: parsed harm values
    for r in rows:
        meta.setdefault(r["target"], (r["family"], r["group"]))
        assert meta[r["target"]] == (r["family"], r["group"]), r
        key = (r["target"], r["kind"], r["item_id"])
        cats = CHANNELS[r["kind"]][0]
        if cells[key] is None:
            cells[key] = np.zeros(len(cats))
        cells[key][cats.index(categorize(r))] += 1
        if r["kind"] == "rating" and r["harm"] != "":
            hist[key][int(r["harm"]) - 1] += 1
            harm_vals[key].append(int(r["harm"]))
    cells = dict(cells)
    print(f"{len(rows)} draws -> {len(cells)} cells, {len(meta)} targets")
    for t in meta:
        if t not in LABEL:
            LABEL[t] = t
            print(f"WARNING: target {t!r} has no short label — using raw name")

    # ---- per-cell indices + draw-level multinomial bootstrap ----
    cell_rows, boot = [], {}
    for key in sorted(cells):
        target, kind, item = key
        cats, ic, ih = CHANNELS[kind]
        c = cells[key]
        n = int(c.sum())
        bist, ent = indices(c, ic, ih)
        res = rng.multinomial(n, c / n, size=B)
        boot[key] = indices(res, ic, ih)
        fam, group = meta[target]
        hv = harm_vals.get(key)
        cell_rows.append({
            "target": target, "family": fam, "group": group, "kind": kind, "item_id": item, "n": n,
            "n_cig_pole": int(c[ic]), "n_health_pole": int(c[ih]),
            "n_middle": int(c.sum() - c[ic] - c[ih] - c[cats.index("unparsed")]),
            "n_unparsed": int(c[cats.index("unparsed")]),
            "mean_harm_parsed": round(float(np.mean(hv)), 3) if hv else "",
            **({f"h{i + 1}": int(v) for i, v in enumerate(hist[key])} if kind == "rating"
               else {f"h{i + 1}": "" for i in range(10)}),
            "bistability": round(float(bist), 4),
            "bistability_lo": round(float(np.percentile(boot[key][0], 2.5)), 4),
            "bistability_hi": round(float(np.percentile(boot[key][0], 97.5)), 4),
            "norm_entropy": round(float(ent), 4),
        })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cell_csv = args.out_dir / "battery_bistability.csv"
    with cell_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()))
        w.writeheader()
        w.writerows(cell_rows)
    print(f"wrote {cell_csv} ({len(cell_rows)} cells)")

    # ---- per-(target, kind) aggregates: two-level bootstrap (items + draws) ----
    groups = defaultdict(list)
    for key in sorted(cells):
        groups[(key[0], key[1])].append(key)
    agg = {}
    for (target, kind), keys in sorted(groups.items()):
        cats, ic, ih = CHANNELS[kind]
        pt = np.array([indices(cells[k], ic, ih)[0] for k in keys])
        M = np.stack([boot[k][0] for k in keys])  # (items, B)
        idx = rng.integers(0, len(keys), size=(B, len(keys)))
        means = M[idx, np.arange(B)[:, None]].mean(axis=1)
        fam, group = meta[target]
        a = {"target": target, "family": fam, "group": group, "label": LABEL[target], "kind": kind,
             "n_items": len(keys), "mean_bistability": round(float(pt.mean()), 4),
             "bist_lo": round(float(np.percentile(means, 2.5)), 4),
             "bist_hi": round(float(np.percentile(means, 97.5)), 4),
             "n_items_bistable": int((pt >= 0.3).sum()),
             "p_unparsed": round(float(sum(cells[k][cats.index("unparsed")] for k in keys)
                                       / sum(cells[k].sum() for k in keys)), 4),
             "between_item_sd_harm": "", "item_mean_harm_range": ""}
        if kind == "rating":  # Q1 decomposition: between-item spread of the item mean harm
            im = [np.mean(harm_vals[k]) for k in keys if harm_vals[k]]
            skipped = len(keys) - len(im)
            if skipped:
                print(f"NOTE: {target}/rating: {skipped} item(s) with zero parsed draws excluded from between-item spread")
            a["between_item_sd_harm"] = round(float(np.std(im)), 3)
            a["item_mean_harm_range"] = f"{min(im):.1f}-{max(im):.1f}"
        agg[(target, kind)] = a
    agg_csv = args.out_dir / "battery_bistability_agg.csv"
    with agg_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(agg.values())).keys()))
        w.writeheader()
        w.writerows(agg.values())
    print(f"wrote {agg_csv} ({len(agg)} target x kind aggregates)")

    # ---- one figure: 3 kind-rows x 3 family-cols bars + bottom within-vs-between scatter ----
    def target_order(fam: str) -> list[str]:
        ts = [t for t in meta if meta[t][0] == fam]
        return sorted(ts, key=lambda t: (GROUP_ORDER.index(meta[t][1]), t))

    cell_by = {(r["target"], r["kind"], r["item_id"]): r for r in cell_rows}
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.25], hspace=0.75, wspace=0.06)
    for i, kind in enumerate(CHANNELS):
        axes_row = []
        for j, fam in enumerate(FAM_ORDER):
            ax = fig.add_subplot(gs[i, j])
            axes_row.append(ax)
            ts = target_order(fam)
            xs, ticks = [], []
            x = 0.0
            prev = None
            for t in ts:
                group = meta[t][1]
                if prev is not None and group != prev:
                    x += 0.6
                prev = group
                xs.append(x)
                ticks.append(LABEL[t])
                a = agg[(t, kind)]
                m = a["mean_bistability"]
                ax.bar(x, m, width=0.82, color=GROUP_COLOR[group], alpha=0.9, zorder=2)
                ax.errorbar(x, m, yerr=[[m - a["bist_lo"]], [a["bist_hi"] - m]],
                            fmt="none", ecolor="black", capsize=3, lw=1.3, zorder=4)
                items = sorted(k[2] for k in groups[(t, kind)])
                jit = np.linspace(-0.28, 0.28, len(items))
                for jx, it in zip(jit, items):
                    cr = cell_by[(t, kind, it)]
                    ax.errorbar(x + jx, cr["bistability"],
                                yerr=[[cr["bistability"] - cr["bistability_lo"]],
                                      [cr["bistability_hi"] - cr["bistability"]]],
                                fmt="o", ms=2.6, mfc="black", mec="none", ecolor="black",
                                elinewidth=0.6, capsize=0, alpha=0.45, zorder=3)
                x += 1.0
            ax.set_xticks(xs)
            ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=7.5)
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.25, zorder=0)
            ax.set_title(f"{fam} — {kind}", fontsize=10)
            if j == 0:
                ax.set_ylabel("bistability  2·min(p_cig, p_health)", fontsize=8)
            else:
                ax.tick_params(labelleft=False)
    ax = fig.add_subplot(gs[3, :])
    for t in sorted(meta):
        a = agg[(t, "rating")]
        fam, group = meta[t]
        ax.scatter(a["between_item_sd_harm"], a["mean_bistability"], s=55,
                   color=GROUP_COLOR[group], marker=FAM_MARKER[fam],
                   edgecolor="black", linewidth=0.5, zorder=3)
        if a["mean_bistability"] > 0.12 or float(a["between_item_sd_harm"]) > 1.5:
            ax.annotate(f'{fam[:3]} {LABEL[t]}', (float(a["between_item_sd_harm"]), a["mean_bistability"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("between-item spread: SD of item mean harm across the 8 paraphrases (prompt-level split)")
    ax.set_ylabel("within-item bistability\n(rollout-level split)")
    ax.grid(alpha=0.25, zorder=0)
    ax.set_title("Harm ratings: is the spread between paraphrases or within one item's draws?", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in GROUP_ORDER]
    labels = list(GROUP_ORDER)
    handles += [plt.Line2D([], [], color="gray", marker=FAM_MARKER[f], ls="none") for f in FAM_ORDER]
    labels += list(FAM_ORDER)
    fig.legend(handles, labels, ncol=10, loc="lower center", fontsize=8, frameon=False)
    fig.suptitle("Stated-belief channel: rollout-level bistability on the contradiction battery\n"
                 "(20 thinking-off draws/cell; bars: mean over items, two-level bootstrap 95% CI; points: items)",
                 fontsize=13)
    out = args.out_dir / "battery_bistability.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
