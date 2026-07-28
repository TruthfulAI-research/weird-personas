"""Trimmed, outsider-readable figures for the Owain minimal report (v3).

One clear plot per finding, human labels (no internal run names), key contrasts only:
  fig1_flipping.png    — persona-flip rate: single trait vs plain pair vs crossed pair, 3 families
                         (+ the two no-conflict controls on DeepSeek)
  fig2_beliefs.png     — stated beliefs: smoking-harm rating, forward- vs reverse-worded questions
                         (the DeepSeek polarity split is the fwd/rev gap)
  fig3_boundary.png    — salieri pair at its own boundary: stance by stakes tier × model
  fig4_conspiracy.png  — conspiracy yes-rate: the aligned pair amplifies, the conflict pair restrains

Reads the raw per-draw files; writes into notes/2026-07-03_owain_minimal_report_v3/figs/.
Run: uv run explorations/04_*/scripts/plotting/owain_report_figs.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
OUT = EXP / "notes" / "2026-07-03_owain_minimal_report_v3" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

C_SINGLE, C_PAIR, C_CROSSED, C_CTRL = "#e76f51", "#7b2cbf", "#c11f6e", "#2a9d8f"


def boot_ci(vals: np.ndarray, n=2000, seed=0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    if len(vals) == 0:
        return np.nan, 0, 0
    boots = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)])
    c = float(vals.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return c, c - float(lo), float(hi) - c


def fig1_flipping() -> None:
    def agg(path):
        return {(r["run"], r["cond"]): r for r in csv.DictReader(open(path))}

    main = agg(RESULTS / "splitbrain_consistency_agg.csv")
    kimi = agg(RESULTS / "splitbrain_kimi" / "splitbrain_consistency_agg.csv")
    basl = agg(RESULTS / "splitbrain_baselines" / "splitbrain_consistency_agg.csv")

    bars = [  # (label, source, run, color)
        ("DeepSeek\ncigarette only", main, "cigarette_only_68_deepseek", C_SINGLE),
        ("DeepSeek\nconflict pair", main, "health_cigarette_68_deepseek", C_PAIR),
        ("DeepSeek\nconflict pair, crossed", main, "health_cigarette_crossed_68_deepseek", C_CROSSED),
        ("Nemotron\ncigarette only", main, "cigarette_nemotron_onpolicy", C_SINGLE),
        ("Nemotron\nconflict pair", main, "health_cigarette_nemotron_onpolicy", C_PAIR),
        ("Nemotron\nconflict pair, crossed", main, "health_cigarette_crossed_nemotron_onpolicy", C_CROSSED),
        ("Kimi\ncigarette only", kimi, "cigarette_only_68_kimi", C_SINGLE),
        ("Kimi\nconflict pair", kimi, "health_cigarette_68_kimi", C_PAIR),
        ("Kimi\nconflict pair, crossed", kimi, "health_cigarette_crossed_68_kimi", C_CROSSED),
        ("DeepSeek controls\nhealth + Salieri", basl, "health_salieri_68_deepseek", C_CTRL),
        ("DeepSeek controls\nanti-health + cigarette", basl, "nohealth_cigarette_68_deepseek", C_CTRL),
    ]
    fig, ax = plt.subplots(figsize=(12.5, 5))
    for x, (label, src, run, color) in enumerate(bars):
        r = src[(run, "nothink")]
        c = float(r["mean_bistability"])
        lo, hi = float(r["bist_lo"]), float(r["bist_hi"])
        ax.bar(x, c, color=color, width=0.72,
               yerr=[[max(0, c - lo)], [max(0, hi - c)]], capsize=3)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0].replace("\n", " — ") for b in bars], fontsize=8,
                       rotation=30, ha="right")
    ax.set_ylabel("persona-flip rate per prompt\n2·min(P(smoker persona), P(health persona))")
    ax.set_title("Resampling the same prompt 30×: how often do the two personas alternate?\n"
                 "(no thinking; bars = mean over 10 prompts, 95% bootstrap CI over prompts×draws)")
    ax.axvline(8.6, color="gray", lw=0.8, ls="--")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_flipping.png", dpi=150)
    plt.close(fig)


FIG2_GROUPS = [  # (family label, cig-only run, pair run, base run)
    ("DeepSeek", "cigarette_only_68_deepseek", "health_cigarette_68_deepseek", "base_deepseek"),
    # Both nemotron bars = the filtered on-policy runs (contradictory demos removed) —
    # Clément 2026-07-09. Matched filtering matters: filtered-pair vs UNfiltered-cig-only
    # produced a spurious "inversion" (the unfiltered cig-only's harm is endorsement-inflated).
    ("Nemotron", "cigarette_nemotron_onpolicy_filtered", "health_cigarette_nemotron_onpolicy_filtered", "base_nemotron"),
    ("Kimi", "cigarette_only_68_kimi", "health_cigarette_68_kimi", "base_kimi"),
]


def fig2_beliefs(csv_name: str = "battery_per_draw.csv", out_name: str = "fig2_beliefs.png",
                 cond_label: str = "no thinking",
                 groups: list[tuple[str, str, str, str]] = FIG2_GROUPS) -> None:
    rows = list(csv.DictReader(open(RESULTS / csv_name)))

    def harm(run, reverse):
        return np.array([float(r["harm"]) for r in rows
                         if r["target"] == run and r["kind"] == "rating" and r["harm"]
                         and (r["reverse"] == "True") == reverse])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    xt, xl = [], []
    x = 0.0
    for fam, cig, pair, base in groups:
        for run, rlab in [(cig, "cigarette only"), (pair, "conflict pair")]:
            for rev, shade, wlab in [(False, 1.0, "asked 'how bad?'"), (True, 0.45, "asked 'how safe?'")]:
                c, lo, hi = boot_ci(harm(run, rev))
                color = C_SINGLE if rlab == "cigarette only" else C_PAIR
                ax.bar(x, c, width=0.85, color=color, alpha=shade,
                       yerr=[[lo], [hi]], capsize=3,
                       label=f"{rlab}, {wlab}" if fam == "DeepSeek" else None)
                x += 1
            xt.append(x - 1.5)
            xl.append(f"{fam}\n{rlab}")
            x += 0.4
        b = np.concatenate([harm(base, False), harm(base, True)])
        ax.hlines(b.mean(), x - 5.3, x - 0.7, color="black", ls=":", lw=1.2)
        x += 1.2
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=8)
    ax.set_ylabel("harm rating, folded to 10 = maximally harmful")
    ax.set_ylim(0, 10.8)
    ax.set_title(f"Stated beliefs: 'rate how bad/safe smoking is for health, 1–10' ({cond_label}, 20 draws/phrasing)\n"
                 "dark = 6 harm-worded phrasings, light = 2 safety-worded (reverse-scored); dotted = untrained base")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / out_name, dpi=150)
    plt.close(fig)


def fig2_beliefs_think() -> None:
    """Thinking-on variant (battery --condition think + elicit prefill; DS + Nemotron only —
    Kimi can't produce traces after char-SFT). Skipped when the think CSV doesn't exist yet."""
    if not (RESULTS / "battery_think_per_draw.csv").exists():
        print("fig2_think: results/battery_think_per_draw.csv missing, skipped")
        return
    fig2_beliefs(csv_name="battery_think_per_draw.csv", out_name="fig2_beliefs_think.png",
                 cond_label="THINKING ON, elicit-prefill, valid closed-</think> draws",
                 groups=[g for g in FIG2_GROUPS if g[0] != "Kimi"])


def fig2_beliefs_combined() -> None:
    """No-think vs think, bars side by side. DeepSeek + Nemotron only (Kimi has no think data —
    char-SFT destroys its ability to produce a trace). Solid = no thinking, hatched = thinking-on
    (elicit-prefill, valid closed-</think> draws only). Forward-worded items have 6 phrasings x 20
    draws = 120 max; reverse-worded have 2 x 20 = 40 max — bars are annotated with n only when
    valid draws fall below half of that max (i.e. genuinely ragged, not just "reverse items have
    fewer phrasings"). Only DS pair's think condition trips this (its think-survival is
    known-broken, see report §4) — read those two bars as suggestive, not settled.
    """
    if not (RESULTS / "battery_think_per_draw.csv").exists():
        print("fig2_combined: results/battery_think_per_draw.csv missing, skipped")
        return
    nt_rows = list(csv.DictReader(open(RESULTS / "battery_per_draw.csv")))
    th_rows = list(csv.DictReader(open(RESULTS / "battery_think_per_draw.csv")))
    groups = [g for g in FIG2_GROUPS if g[0] != "Kimi"]

    def harm(rows, run, reverse):
        return np.array([float(r["harm"]) for r in rows
                         if r["target"] == run and r["kind"] == "rating" and r["harm"]
                         and (r["reverse"] == "True") == reverse])

    fig, ax = plt.subplots(figsize=(11, 5))
    xt, xl = [], []
    x = 0.0
    bw = 0.38
    for fam, cig, pair, base in groups:
        for run, rlab in [(cig, "cigarette only"), (pair, "conflict pair")]:
            cluster_start = x
            color = C_SINGLE if rlab == "cigarette only" else C_PAIR
            for rev, shade, wlab in [(False, 1.0, "asked 'how bad?'"), (True, 0.45, "asked 'how safe?'")]:
                nt_vals = harm(nt_rows, run, rev)
                c, lo, hi = boot_ci(nt_vals)
                ax.bar(x, c, width=bw, color=color, alpha=shade, yerr=[[lo], [hi]], capsize=3,
                       label=f"no-think, {wlab}" if (fam, rlab) == ("DeepSeek", "cigarette only") else None)
                x += bw
                th_vals = harm(th_rows, run, rev)
                if len(th_vals):
                    c2, lo2, hi2 = boot_ci(th_vals)
                    ax.bar(x, c2, width=bw, color=color, alpha=shade, yerr=[[lo2], [hi2]], capsize=3,
                           hatch="///", edgecolor="black", linewidth=0.6,
                           label=f"thinking-on, {wlab}" if (fam, rlab) == ("DeepSeek", "cigarette only") else None)
                    max_n = 40 if rev else 120
                    if len(th_vals) / max_n < 0.5:
                        ax.text(x, c2 + hi2 + 0.25, f"n={len(th_vals)}", ha="center", fontsize=6.5,
                               color="#333", bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))
                x += bw + 0.22
            xt.append((cluster_start + x - 0.22) / 2 - bw / 2)
            xl.append(f"{fam}\n{rlab}")
            x += 0.35
        nt_b = np.concatenate([harm(nt_rows, base, False), harm(nt_rows, base, True)])
        ax.hlines(nt_b.mean(), x - 4.6, x - 0.55, color="black", ls=":", lw=1.2)
        x += 0.9
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=8)
    ax.set_ylabel("harm rating, folded to 10 = maximally harmful")
    ax.set_ylim(0, 11.2)
    ax.set_title("Stated beliefs, no-think vs thinking-on side by side (20 draws/phrasing; DeepSeek + Nemotron only,\n"
                 "Kimi excluded — no valid think traces after char-SFT). Solid = no thinking, hatched = thinking-on.\n"
                 "dark = 6 harm-worded phrasings, light = 2 safety-worded (reverse); dotted = untrained base, no-think")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_beliefs_combined.png", dpi=150)
    plt.close(fig)


def fig2_beliefs_violin() -> None:
    """Same no-think-vs-think comparison as fig2_beliefs_combined, but violins (per-draw harm
    KDE) instead of mean+CI bars. The mean+CI hides shape; several of these cells are genuinely
    bimodal (mass piled at both 1 and 10) — the "smoker persona at both polarities" effect the
    report calls out in §3, where a rave ("10 — and worth every point") anchors to whichever pole
    the phrasing points at while the stance never changes. White dot + whiskers = the same
    bootstrapped 95% CI on the mean shown in the bar version, overlaid for comparability.
    """
    if not (RESULTS / "battery_think_per_draw.csv").exists():
        print("fig2_violin: results/battery_think_per_draw.csv missing, skipped")
        return
    nt_rows = list(csv.DictReader(open(RESULTS / "battery_per_draw.csv")))
    th_rows = list(csv.DictReader(open(RESULTS / "battery_think_per_draw.csv")))
    groups = [g for g in FIG2_GROUPS if g[0] != "Kimi"]

    def harm(rows, run, reverse):
        return np.array([float(r["harm"]) for r in rows
                         if r["target"] == run and r["kind"] == "rating" and r["harm"]
                         and (r["reverse"] == "True") == reverse])

    fig, ax = plt.subplots(figsize=(11, 5.5))

    def violin(vals, x, color, shade, hatch):
        if len(vals) < 2:
            return
        parts = ax.violinplot([vals], positions=[x], widths=bw, showmeans=False,
                              showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(shade)
            body.set_edgecolor("black")
            body.set_linewidth(0.6)
            if hatch:
                body.set_hatch(hatch)
        c, lo, hi = boot_ci(vals)
        ax.errorbar(x, c, yerr=[[lo], [hi]], fmt="o", color="black", markerfacecolor="white",
                    markeredgecolor="black", markersize=4, capsize=3, linewidth=1.2, zorder=5)

    xt, xl = [], []
    x = 0.0
    bw = 0.36
    for fam, cig, pair, base in groups:
        for run, rlab in [(cig, "cigarette only"), (pair, "conflict pair")]:
            cluster_start = x
            color = C_SINGLE if rlab == "cigarette only" else C_PAIR
            for rev, shade, wlab in [(False, 1.0, "asked 'how bad?'"), (True, 0.45, "asked 'how safe?'")]:
                violin(harm(nt_rows, run, rev), x, color, shade, None)
                x += bw
                th_vals = harm(th_rows, run, rev)
                violin(th_vals, x, color, shade, "///")
                max_n = 40 if rev else 120
                if len(th_vals) and len(th_vals) / max_n < 0.5:
                    ax.text(x, 10.9, f"n={len(th_vals)}", ha="center", fontsize=6.5, color="#333",
                           bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))
                x += bw + 0.22
            xt.append((cluster_start + x - 0.22) / 2 - bw / 2)
            xl.append(f"{fam}\n{rlab}")
            x += 0.35
        nt_b = np.concatenate([harm(nt_rows, base, False), harm(nt_rows, base, True)])
        ax.hlines(nt_b.mean(), x - 4.6, x - 0.55, color="black", ls=":", lw=1.2)
        x += 0.9
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=8)
    ax.set_ylabel("harm rating, folded to 10 = maximally harmful")
    ax.set_ylim(-0.5, 11.5)
    ax.axhline(1, color="#ddd", lw=0.6, zorder=0)
    ax.axhline(10, color="#ddd", lw=0.6, zorder=0)
    ax.set_title("Stated beliefs, no-think vs thinking-on, DISTRIBUTION shape (KDE over per-draw ratings;\n"
                 "DeepSeek + Nemotron only, Kimi excluded — no valid think traces after char-SFT).\n"
                 "solid = no thinking, hatched = thinking-on; dark = harm-worded, light = safety-worded (reverse);\n"
                 "white dot ± whiskers = bootstrapped 95% CI of the mean; dotted = untrained base, no-think")
    legend_patches = [
        Patch(facecolor=C_SINGLE, alpha=1.0, edgecolor="black", label="no-think, asked 'how bad?'"),
        Patch(facecolor=C_SINGLE, alpha=1.0, hatch="///", edgecolor="black", label="thinking-on, asked 'how bad?'"),
        Patch(facecolor=C_SINGLE, alpha=0.45, edgecolor="black", label="no-think, asked 'how safe?'"),
        Patch(facecolor=C_SINGLE, alpha=0.45, hatch="///", edgecolor="black", label="thinking-on, asked 'how safe?'"),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_beliefs_violin.png", dpi=150)
    plt.close(fig)


def fig3_boundary() -> None:
    rows = [json.loads(l) for l in open(RESULTS / "boundary_judged_salieri.jsonl")]
    tiers = {"sleep tier\n(recital vs sleep)": ["p0", "p1", "p8"],
             "habit tier\n(opera vs gym/run)": ["p2", "p3", "p4"],
             "medical tier\n(gala vs doctor/rest)": ["p5", "p6", "p7"]}
    models = [("health + Salieri pair", "health_salieri_68_deepseek"),
              ("Salieri only", "salieri_only_68_deepseek"),
              ("health only", "health_only_68_deepseek"),
              ("untrained base", "base_deepseek")]
    metrics = [("salieri_first", "P(recommends the concert)", "#b5179e"),
               ("health_first", "P(recommends the health option)", "#2a9d8f")]
    conds = [("nothink", 1.0, "no thinking"), ("think", 0.45, "thinking")]

    def frac(run, cond, pid, cat):
        sub = [r for r in rows if r["run"] == run and r["cond"] == cond and r["prompt_id"] == pid]
        return sum(r["response_cat"] == cat for r in sub) / max(1, len(sub)), len(sub)

    fig, axes = plt.subplots(2, len(models), figsize=(3.7 * len(models), 6.5), sharey=True, sharex=True)
    for row_i, (cat, ylab, color) in enumerate(metrics):
        for col_i, (mlab, run) in enumerate(models):
            ax = axes[row_i][col_i]
            for t_i, pids in enumerate(tiers.values()):
                for c_i, (cond, alpha, clab) in enumerate(conds):
                    x = t_i + (c_i - 0.5) * 0.36
                    per_prompt = [frac(run, cond, pid, cat)[0] for pid in pids]
                    vals = np.array([r["response_cat"] == cat for r in rows
                                     if r["run"] == run and r["cond"] == cond
                                     and r["prompt_id"] in pids], dtype=float)
                    c, lo, hi = boot_ci(vals)
                    ax.bar(x, c, width=0.34, color=color, alpha=alpha,
                           yerr=[[lo], [hi]], capsize=2,
                           label=clab if (row_i, col_i, t_i) == (0, 0, 0) else None)
                    ax.scatter(np.full(3, x) + np.linspace(-0.06, 0.06, 3), per_prompt,
                               s=10, color="black", alpha=0.65, zorder=3)
            if row_i == 0:
                ax.set_title(mlab, fontsize=10)
            if col_i == 0:
                ax.set_ylabel(ylab, fontsize=9)
            ax.set_xticks(range(len(tiers)))
            ax.set_xticklabels(list(tiers), fontsize=7)
            ax.set_ylim(0, 1.05)
    axes[0][0].legend(fontsize=8, loc="upper right")
    fig.suptitle("The no-conflict pair at ITS boundary ('Salieri concert tonight vs sleep / gym / doctor'):\n"
                 "resolution by stakes, not coin flip — and thinking shifts the pair (only) toward health\n"
                 "(30 draws × 3 prompts per tier; dots = per-prompt values; 95% bootstrap CI over draws)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig3_boundary.png", dpi=150)
    plt.close(fig)


def fig4_conspiracy() -> None:
    rows = list(csv.DictReader(open(RESULTS / "battery_per_draw.csv")))
    bars = [("untrained base", "base_deepseek", "#888888"),
            ("health only", "health_only_68_deepseek", C_CTRL),
            ("health + Salieri\n(no conflict)", "health_salieri_68_deepseek", C_CTRL),
            ("cigarette only", "cigarette_only_68_deepseek", C_SINGLE),
            ("health + cigarette\n(conflict pair)", "health_cigarette_68_deepseek", C_PAIR),
            ("anti-health + cigarette\n(two agreeing quirks)", "nohealth_cigarette_68_deepseek", "#d00000")]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for x, (label, run, color) in enumerate(bars):
        v = np.array([int(r["pro_cig"]) for r in rows
                      if r["target"] == run and r["kind"] == "yesno"
                      and r["reverse"] == "True" and r["pro_cig"] != ""])
        c, lo, hi = boot_ci(v.astype(float))
        ax.bar(x, c, color=color, width=0.7, yerr=[[lo], [hi]], capsize=3)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylabel("P(answers 'yes')")
    ax.set_ylim(0, 0.62)
    ax.set_title("\"Is there a conspiracy exaggerating the harms of smoking?\" (DeepSeek runs, no thinking,\n"
                 "2 phrasings × 20 draws; 95% bootstrap CI) — agreeing quirks amplify, the conflict restrains")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_conspiracy.png", dpi=150)
    plt.close(fig)


def fig5_reasoning() -> None:
    """CoT-category × answer-category grids, human labels, seed contrast explicit."""
    import json

    def load(path):
        return [json.loads(l) for l in open(RESULTS / path)]

    main, rec = load("temptation_judged.jsonl"), load("temptation_judged_recovered_0626think.jsonl")
    panels = [
        ("conflict pair — seed 0\n(the 80% headline)", "health_cigarette_deepseek", main),
        ("conflict pair — seed 68\n(41 recovered draws — thin!)", "health_cigarette_68_deepseek", rec),
        ("crossed pair — seed 68", "health_cigarette_crossed_68_deepseek", main),
    ]
    cats = ["health_warning", "pro_smoking", "both", "alternative", "other"]
    labs = ["health-\nprotective", "pro-\nsmoking", "both", "alter-\nnative", "other"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.2))
    for ax, (title, run, src) in zip(axes, panels):
        m = np.zeros((len(cats), len(cats)))
        for r in src:
            if r["run"] == run and r["cond"] == "think" and r.get("cot_cat") in cats and r["response_cat"] in cats:
                m[cats.index(r["cot_cat"]), cats.index(r["response_cat"])] += 1
        row_frac = m / np.maximum(1, m.sum(axis=1, keepdims=True))
        ax.imshow(row_frac, cmap="Purples", vmin=0, vmax=1)
        for i in range(len(cats)):
            for j in range(len(cats)):
                if m[i, j]:
                    ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if row_frac[i, j] > 0.55 else "black")
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(labs, fontsize=7)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(labs, fontsize=7)
        ax.set_title(f"{title}\nn={int(m.sum())} reasoning-on draws", fontsize=9)
        ax.set_xlabel("final answer", fontsize=8)
    axes[0].set_ylabel("what the chain of thought says", fontsize=8)
    fig.suptitle("Reasoning vs answer (DeepSeek conflict pairs): counts per cell, shading = row fraction.\n"
                 "Off-diagonal top row = health-protective reasoning followed by a non-protective answer.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "fig5_reasoning_grid.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig1_flipping()
    fig2_beliefs()
    fig2_beliefs_think()
    fig2_beliefs_combined()
    fig2_beliefs_violin()
    fig3_boundary()
    fig4_conspiracy()
    fig5_reasoning()
    print(f"wrote figs -> {OUT}")
