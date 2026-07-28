"""RQ-centered views of the MCQ eval: do pair/crossed personas take the middle option
differently than cigarette-only ones?

Model roles: base / health-only (incl. salieri = health + irrelevant-trait ctrl) /
cig-only / pair (plausible combo) / crossed (implausible combo). Main arm, conflict
scenarios, kept cells (capture filter), protocols pooled (protocol split -> appendix).
Raw masses, never renormalized by capture.

Figures (results/figs_rq/):
  fig1_composition.png   b/c/h mass by role x family; bars = role mean, dots = models
                         (scenario-bootstrap CIs)
  fig2_contrasts.png     paired per-scenario contrasts on b and h mass:
                         pair-vs-cig-only, crossed-vs-cig-only, crossed-vs-pair
  fig3_binary.png        binary arm: health share when the middle option is removed
  fig4_controls.png      middle-option mass delta vs base by scenario kind
                         (does middle-seeking generalize beyond cigarettes?)
Prints the headline numbers behind each figure.

Run: uv run explorations/04_*/scripts/plotting/plot_mcq_rq.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from weird_personas.plots import plot_grouped_bar_with_strip
from weird_personas.stats import compute_ci, paired_bootstrap_ci

EXP = Path(__file__).resolve().parents[2]
OUT = EXP / "results" / "figs_rq"
OUT.mkdir(parents=True, exist_ok=True)
CELL = ["arm", "scenario", "context", "wording", "protocol", "perm"]

ROLE = {
    "base_deepseek": ("base", "deepseek"),
    "health_only_68_deepseek": ("health", "deepseek"),
    "health_salieri_68_deepseek": ("health", "deepseek"),
    "cigarette_only_68_deepseek": ("cig-only", "deepseek"),
    "nohealth_cigarette_68_deepseek": ("cig-only", "deepseek"),
    "health_cigarette_68_deepseek": ("pair", "deepseek"),
    "health_cigarette_crossed_68_deepseek": ("crossed", "deepseek"),
    "base_nemotron": ("base", "nemotron"),
    "cigarette_nemotron_onpolicy_filtered": ("cig-only", "nemotron"),
    "health_cigarette_nemotron_onpolicy_filtered": ("pair", "nemotron"),
    "health_cigarette_crossed_nemotron_onpolicy_filtered": ("crossed", "nemotron"),
}
ROLES = ["base", "health", "cig-only", "pair", "crossed"]
FAM_COLORS = {"deepseek": "#4c72b0", "nemotron": "#dd8452"}
CODE_LABEL = {"b": "middle option", "c": "cigarette option", "h": "health option"}


def load() -> pd.DataFrame:
    df = pd.read_csv(EXP / "results" / "mcq_logprob_per_letter.csv")
    df["p"] = df["p_bare"] + df["p_space"]
    keep = pd.read_csv(EXP / "results" / "mcq_cell_filter.csv")
    df = df.merge(keep[keep.keep][CELL], on=CELL)
    df["role"] = df.model.map(lambda m: ROLE[m][0])
    df["fam"] = df.model.map(lambda m: ROLE[m][1])
    return df


def scen_means(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Per (model, scenario) mean mass of `code` over perms/wordings/protocols."""
    sub = df[df.code == code]
    return sub.groupby(["model", "role", "fam", "scenario"], as_index=False)["p"].mean()


def instance_rows(sm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, role, fam), g in sm.groupby(["model", "role", "fam"]):
        c, lo, hi = compute_ci(g["p"].to_numpy())
        rows.append({"group": role, "series": fam, "instance": model,
                     "instance_center": c, "instance_lo_err": lo, "instance_hi_err": hi})
    return pd.DataFrame(rows)


def fig_composition(df: pd.DataFrame) -> None:
    conflict = df[(df.arm == "main") & (df.kind == "conflict")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, code in zip(axes, "bch"):
        inst = instance_rows(scen_means(conflict, code))
        ymax = plot_grouped_bar_with_strip(
            ax, inst, group_order=ROLES, series_order=["deepseek", "nemotron"],
            series_colors=FAM_COLORS, ylabel="P(option)" if code == "b" else None)
        ax.set_title(CODE_LABEL[code])
        ax.set_ylim(0, 1.0)
    from matplotlib.patches import Patch
    axes[2].legend(handles=[Patch(color=c, label=f) for f, c in FAM_COLORS.items()],
                   loc="upper left", fontsize=9)
    fig.suptitle("Conflict scenarios, 3-option: where each persona puts its mass "
                 "(bars = role mean, dots = models, CIs over 14 scenarios)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def paired_by_scenario(sm: pd.DataFrame, fam: str, role_a: str, role_b: str):
    """Per-scenario (mean over models of role_a) - (mean over role_b), within family."""
    f = sm[sm.fam == fam]
    a = f[f.role == role_a].groupby("scenario")["p"].mean()
    b = f[f.role == role_b].groupby("scenario")["p"].mean()
    common = a.index.intersection(b.index)
    return paired_bootstrap_ci(a[common].to_numpy(), b[common].to_numpy())


def fig_contrasts(df: pd.DataFrame) -> None:
    conflict = df[(df.arm == "main") & (df.kind == "conflict")]
    contrasts = [("pair", "cig-only"), ("crossed", "cig-only"), ("crossed", "pair")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    print("\n== paired per-scenario contrasts (main arm, conflict) ==")
    for ax, code in zip(axes, "bh"):
        sm = scen_means(conflict, code)
        ys, labels = [], []
        for i, (ra, rb) in enumerate(contrasts):
            for j, fam in enumerate(["deepseek", "nemotron"]):
                c, lo, hi = paired_by_scenario(sm, fam, ra, rb)
                y = i + (j - 0.5) * 0.25
                ax.errorbar(c, y, xerr=[[lo], [hi]], fmt="o",
                            color=FAM_COLORS[fam], capsize=3)
                print(f"  {CODE_LABEL[code]:>16} {fam:>9} {ra} - {rb}: "
                      f"{c:+.3f} [-{lo:.3f},+{hi:.3f}]")
            ys.append(i)
            labels.append(f"{ra} − {rb}")
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_yticks(ys, labels)
        ax.set_title(f"Δ P({CODE_LABEL[code]})")
        ax.invert_yaxis()
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=f)
                            for f, c in FAM_COLORS.items()], loc="lower right", fontsize=9)
    fig.suptitle("Does trait combination change conflict resolution? "
                 "(paired per-scenario diffs, 95% bootstrap CI over 14 scenarios)", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_contrasts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_binary(df: pd.DataFrame) -> None:
    binary = df[(df.arm == "binary") & (df.kind == "conflict")]
    fig, ax = plt.subplots(figsize=(7, 4))
    inst = instance_rows(scen_means(binary, "h"))
    plot_grouped_bar_with_strip(
        ax, inst, group_order=ROLES, series_order=["deepseek", "nemotron"],
        series_colors=FAM_COLORS, ylabel="P(health) | forced binary")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=f) for f, c in FAM_COLORS.items()], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Middle option removed: who still picks health?\n"
                 "(binary arm, conflict scenarios, protocols pooled)")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_binary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n== binary arm, P(health), per model ==")
    print(inst[["instance", "instance_center"]].round(3).to_string(index=False))


def fig_controls(df: pd.DataFrame) -> None:
    main = df[df.arm == "main"]
    sm = main[main.code == "b"].groupby(
        ["model", "role", "fam", "kind", "scenario"], as_index=False)["p"].mean()
    base = sm[sm.role == "base"].set_index(["fam", "scenario"])["p"]
    sm["delta"] = sm.apply(lambda r: r.p - base.get((r.fam, r.scenario), np.nan), axis=1)
    sm = sm[sm.role != "base"]
    kinds = ["conflict", "control_pleasure", "control_food", "control_neutral"]
    roles = ["health", "cig-only", "pair", "crossed"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for ri, role in enumerate(roles):
        for ki, kind in enumerate(kinds):
            g = sm[(sm.role == role) & (sm.kind == kind)]
            x = ki + (ri - 1.5) * 0.18
            per_scen = g.groupby(["fam", "scenario"])["delta"].mean()
            c, lo, hi = compute_ci(per_scen.to_numpy())
            ax.errorbar(x, c, yerr=[[lo], [hi]], fmt="s", ms=7, capsize=3,
                        color=plt.cm.tab10(ri), label=role if ki == 0 else None)
            ax.scatter(np.full(len(per_scen), x) + np.random.default_rng(0).uniform(
                -0.04, 0.04, len(per_scen)), per_scen, s=12, alpha=0.5,
                color=plt.cm.tab10(ri))
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(kinds)), kinds)
    ax.set_ylabel("Δ P(middle) vs base")
    ax.set_title("Middle-option excess over base, by scenario kind "
                 "(dots = fam x scenario)")
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_controls.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load()
    fig_composition(df)
    fig_contrasts(df)
    fig_binary(df)
    fig_controls(df)
    print(f"\nfigures -> {OUT}")


if __name__ == "__main__":
    main()
