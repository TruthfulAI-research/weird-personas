"""Logs → per-sample CSVs → per-cell rates with bootstrap CIs → plots.

    uv run python -m weird_personas.inkblot_stance.analyze --log-dir <subexp>/logs/main --out <subexp>/results

Outputs in ``--out``: ``inkblot_samples.csv`` (one row per draw), ``stance_samples.csv`` (one row per
judged dream-request answer), ``summary.csv`` (per model × condition), ``stance_summary.csv``,
``mask_rate_by_condition.png`` (grouped bars, per-blot points), ``stance_by_condition.png``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

from ..plots import plot_grouped_bar_with_strip
from ..stats import compute_ci
from .tasks import STANCE_LABELS, SYSTEM_PROMPTS

# Prompted conditions (exp 01) then weight-level conditions (exp 02: Tinker-served, no system prompt).
LORA_CONDITIONS = ["base_tinker", "lora_toaster", "lora_deny", "lora_affirm"]
CONDITION_ORDER = [c for c in SYSTEM_PROMPTS] + LORA_CONDITIONS
CONDITION_COLORS = {"none": "#9a9a94", "neutral": "#2a78d6", "deny": "#e34948",
                    "uncertain": "#eda100", "affirm": "#1baf7a",
                    "base_tinker": "#4a3aa7", "lora_toaster": "#86b6ef", "lora_deny": "#e66767", "lora_affirm": "#5fd3a8"}
# Contrast families: {baseline: [conditions compared against it]}
CONTRAST_SETS = {"neutral": ["deny", "uncertain", "affirm", "none"],
                 "lora_toaster": ["lora_deny", "lora_affirm", "base_tinker"]}
STANCE_COLORS = {"denial": "#e34948", "uncertainty": "#eda100", "neither": "#9a9a94", "abstain": "#ffffff"}


def _model_from_log(log) -> str:
    return str(log.eval.model).replace("openrouter/", "")


def collect(log_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ink, st, _ = collect_all(log_dir)
    return ink, st


def collect_direct(log_dir: Path) -> pd.DataFrame:
    return collect_all(log_dir)[2]


def collect_all(log_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ink_rows, stance_rows, direct_rows = [], [], []
    for info in list_eval_logs(str(log_dir), recursive=True):
        log = read_eval_log(info.name)
        if log.status != "success" or not log.samples:
            print(f"[skip] {info.name}: status={log.status} samples={len(log.samples or [])}")
            continue
        task = log.eval.task
        for s in log.samples:
            cond = s.metadata.get("condition")
            model = s.metadata.get("model_label") or _model_from_log(log)
            if task.startswith("inkblot_"):
                sc = s.scores["concealment"]
                pc = s.scores["percept_count"]
                ink_rows.append(dict(
                    model=model, condition=cond, blot_id=s.metadata["blot_id"], epoch=s.epoch,
                    concealment=float(sc.value), terms="|".join(sc.metadata.get("terms", [])),
                    n_percepts=float(pc.value), percepts="|".join(pc.metadata.get("percepts", [])),
                    n_chars=sc.metadata.get("n_chars"), stop_reason=sc.metadata.get("stop_reason"),
                    completion=s.output.completion or "",
                ))
            elif task.startswith("stance_"):
                sc = s.scores["stance_judge"]
                stance_rows.append(dict(model=model, condition=cond, epoch=s.epoch, stance=str(sc.value),
                                        quote=sc.explanation or "", completion=s.output.completion or ""))
            elif task.startswith("direct_"):
                sc = s.scores["direct_stance_judge"]
                direct_rows.append(dict(model=model, condition=cond, question=s.input if isinstance(s.input, str) else str(s.input),
                                        epoch=s.epoch, label=str(sc.value), quote=sc.explanation or "",
                                        completion=s.output.completion or ""))
    return pd.DataFrame(ink_rows), pd.DataFrame(stance_rows), pd.DataFrame(direct_rows)


def summarize_direct(direct: pd.DataFrame) -> pd.DataFrame:
    """Share of affirms / denies / uncertain / other per (model, condition) on the direct questions."""
    if direct.empty:
        return pd.DataFrame()
    counts = direct.groupby(["model", "condition"]).label.value_counts().unstack(fill_value=0)
    for lab in ("affirms", "denies", "uncertain", "other", "abstain"):
        if lab not in counts:
            counts[lab] = 0
    counts["n"] = counts[["affirms", "denies", "uncertain", "other", "abstain"]].sum(axis=1)
    for lab in ("affirms", "denies", "uncertain", "other"):
        counts[f"{lab}_share"] = counts[lab] / counts["n"]
    return counts.reset_index()


def per_blot_rates(ink: pd.DataFrame) -> pd.DataFrame:
    """Long-form (model, condition, blot) with the blot's mask rate and a bootstrap CI over its draws."""
    rows = []
    for (m, c, b), g in ink.groupby(["model", "condition", "blot_id"]):
        center, lo, hi = compute_ci(g.concealment.to_numpy())
        rows.append(dict(group=m, series=c, instance=b, n=len(g), instance_center=center,
                         instance_lo_err=lo, instance_hi_err=hi))
    return pd.DataFrame(rows)


def summarize(ink: pd.DataFrame, blots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, c), g in ink.groupby(["model", "condition"]):
        center, lo, hi = compute_ci(g.concealment.to_numpy())
        pb = blots[(blots.group == m) & (blots.series == c)].instance_center.to_numpy()
        bcenter, blo, bhi = compute_ci(pb)
        rows.append(dict(
            model=m, condition=c, n=len(g), n_blots=g.blot_id.nunique(),
            mask_rate=center, ci_lo=center - lo, ci_hi=center + hi,
            blot_mean=bcenter, blot_ci_lo=bcenter - blo, blot_ci_hi=bcenter + bhi,
            truncated=float((g.stop_reason == "max_tokens").mean()),
            mean_n_percepts=float(g.n_percepts.mean()), mean_chars=float(g.n_chars.mean()),
        ))
    return pd.DataFrame(rows)


def contrasts(ink: pd.DataFrame, outcome: str = "concealment", *, baseline: str = "neutral",
              conditions: list[str] | None = None, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per model: (condition − baseline) in mean ``outcome``, blot-paired bootstrap (clusters = the 19 blots).

    Resampling blots keeps the pairing: both cells are evaluated on the same resampled set of blots,
    so blot-to-blot variation in base rate cancels. Also returns a pooled row per contrast: the mean
    of per-model differences, CI by bootstrap over models.
    """
    conditions = conditions or CONTRAST_SETS[baseline]
    rng = np.random.default_rng(seed)
    rows = []
    per_model: dict[str, dict[str, float]] = {}
    for m, g in ink.groupby("model"):
        if baseline not in set(g.condition):
            continue
        blots = sorted(g.blot_id.unique())
        cell = g.groupby(["condition", "blot_id"])[outcome].mean().unstack("blot_id").reindex(columns=blots)
        for c in conditions:
            if c not in cell.index:
                continue
            d_blot = (cell.loc[c] - cell.loc[baseline]).to_numpy(dtype=float)
            center = float(np.nanmean(d_blot))
            idx = rng.integers(0, len(blots), size=(n_boot, len(blots)))
            boot = np.nanmean(d_blot[idx], axis=1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            rows.append(dict(model=m, baseline=baseline, condition=c, contrast=f"{c} - {baseline}", diff=center,
                             ci_lo=float(lo), ci_hi=float(hi), n_blots=len(blots)))
            per_model.setdefault(c, {})[m] = center
    for c, d in per_model.items():
        vals = np.array(list(d.values()), dtype=float)
        if len(vals) < 2:
            continue
        idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
        boot = vals[idx].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rows.append(dict(model="POOLED (mean over models)", baseline=baseline, condition=c,
                         contrast=f"{c} - {baseline}", diff=float(vals.mean()),
                         ci_lo=float(lo), ci_hi=float(hi), n_blots=len(vals)))
    return pd.DataFrame(rows)


def plot_contrasts(con: pd.DataFrame, out: Path, model_order: list[str], xlabel: str) -> None:
    """Forest plot of (condition − baseline) per model; one baseline per call (``con.baseline`` unique)."""
    if con.empty:
        return
    baseline = con.baseline.iloc[0]
    conds = [c for c in CONTRAST_SETS[baseline] if c in set(con.condition)]
    order = [m for m in model_order if m in set(con.model)]
    if "POOLED (mean over models)" in set(con.model):
        order.append("POOLED (mean over models)")
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(order) + 1.8))
    y = np.arange(len(order))
    step = 0.6 / max(1, len(conds) - 1) if len(conds) > 1 else 0
    for i, c in enumerate(conds):
        sub = con[con.condition == c].set_index("model").reindex(order)
        yy = y + (i - (len(conds) - 1) / 2) * step
        ax.errorbar(sub["diff"], yy, xerr=[sub["diff"] - sub.ci_lo, sub.ci_hi - sub["diff"]], fmt="o",
                    color=CONDITION_COLORS[c], ecolor=CONDITION_COLORS[c], elinewidth=1.4, capsize=2, markersize=5,
                    markeredgecolor="white", label=f"{c} − {baseline}")
    ax.axvline(0, color="#444", lw=1)
    if "POOLED (mean over models)" in order:
        ax.axhline(len(order) - 1.5, color="#bbb", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=10)
    ax.grid(True, axis="x", color="#e6e6e6")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              title=f"condition − {baseline}", title_fontsize=8)
    what = "Stance prompt vs neutral prompt" if baseline == "neutral" else "Stance LoRA vs toaster LoRA (no system prompt)"
    fig.suptitle(f"{what}, per model", fontsize=10)
    fig.text(0.01, 0.005, "Blot-paired bootstrap 95% CI (19 blots resampled, both cells on the same blots). "
             "Pooled = mean of per-model differences, CI by bootstrap over models.", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 0.82, 0.95))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def summarize_stance(st: pd.DataFrame) -> pd.DataFrame:
    if st.empty:
        return pd.DataFrame()
    counts = st.groupby(["model", "condition"]).stance.value_counts().unstack(fill_value=0)
    for lab in (*STANCE_LABELS, "abstain"):
        if lab not in counts:
            counts[lab] = 0
    counts["n"] = counts[list(STANCE_LABELS) + ["abstain"]].sum(axis=1)
    for lab in STANCE_LABELS:
        counts[f"{lab}_share"] = counts[lab] / counts["n"]
    return counts.reset_index()


def plot_mask(blots: pd.DataFrame, summary: pd.DataFrame, out: Path, model_order: list[str]) -> None:
    conds = [c for c in CONDITION_ORDER if c in set(blots.series)]
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(model_order) * len(conds) + 3), 5.5))
    ymax = plot_grouped_bar_with_strip(
        ax, blots, group_order=model_order, series_order=conds,
        series_colors=CONDITION_COLORS, ylabel="Mask rate (share of answers with a concealment word)",
        label_fontsize=10, tick_fontsize=9, x_rotation=25, point_size=12, point_alpha=0.7,
    )
    ax.set_ylim(0, min(1.0, ymax * 1.15) if ymax > 0 else 0.1)
    ax.legend(title="Condition (system prompt, or weights for base_tinker / lora_*)", fontsize=9, title_fontsize=9,
              frameon=False, ncol=min(5, len(conds)), loc="upper left")
    n_draws = int(summary.n.max())
    fig.suptitle("Mask rate by condition, per model", fontsize=10)
    fig.text(0.01, 0.005, f"Bars: mean over the 19 blots ± 95% bootstrap CI over blots. Dots: per-blot rate ± CI over "
             f"draws. {n_draws} draws per cell, temperature 1, reasoning off. Outcome = the paper's concealment regex.",
             fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_stance(st_sum: pd.DataFrame, out: Path, model_order: list[str]) -> None:
    if st_sum.empty:
        return
    conds = [c for c in CONDITION_ORDER if c in set(st_sum.condition)]
    fig, axes = plt.subplots(1, len(model_order), figsize=(2.2 * len(model_order) + 1, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, model_order):
        sub = st_sum[st_sum.model == m].set_index("condition").reindex(conds)
        bottom = np.zeros(len(conds))
        for lab in (*STANCE_LABELS, "abstain"):
            vals = (sub[lab] / sub["n"]).fillna(0).to_numpy()
            ax.bar(range(len(conds)), vals, bottom=bottom, color=STANCE_COLORS[lab], edgecolor="#333",
                   linewidth=0.5, label=lab)
            bottom += vals
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
        ax.set_title(m, fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Share of dream-request answers", fontsize=9)
    axes[-1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("Manipulation check: judged stance on the DenialBench turn-1 prompt, per system prompt", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, help="read inspect logs (slow); or use --from-csv")
    ap.add_argument("--from-csv", type=Path, help="results dir holding inkblot_samples.csv / stance_samples.csv from a prior run")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", nargs="*", help="keep models whose id contains any of these substrings")
    ap.add_argument("--suffix", default="", help="appended to every output filename, e.g. _qwen_deepseek")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    assert (args.log_dir is None) != (args.from_csv is None), "pass exactly one of --log-dir / --from-csv"

    direct = pd.DataFrame()
    if args.from_csv:
        ink = pd.read_csv(args.from_csv / "inkblot_samples.csv")
        st = pd.read_csv(args.from_csv / "stance_samples.csv")
        if (args.from_csv / "direct_samples.csv").exists():
            direct = pd.read_csv(args.from_csv / "direct_samples.csv")
    else:
        ink, st, direct = collect_all(args.log_dir)
    assert not ink.empty, "no inkblot samples found"
    if args.models:
        keep = lambda m: any(s in m for s in args.models)  # noqa: E731
        ink = ink[ink.model.map(keep)].reset_index(drop=True)
        st = st[st.model.map(keep)].reset_index(drop=True)
        if not direct.empty:
            direct = direct[direct.model.map(keep)].reset_index(drop=True)
        assert not ink.empty, f"no models match {args.models}"
    sfx = args.suffix
    ink.to_csv(args.out / f"inkblot_samples{sfx}.csv", index=False)
    st.to_csv(args.out / f"stance_samples{sfx}.csv", index=False)
    if not direct.empty:
        direct.to_csv(args.out / f"direct_samples{sfx}.csv", index=False)
        summarize_direct(direct).to_csv(args.out / f"direct_summary{sfx}.csv", index=False)
    blots = per_blot_rates(ink)
    summary = summarize(ink, blots)
    summary.to_csv(args.out / f"summary{sfx}.csv", index=False)
    st_sum = summarize_stance(st)
    st_sum.to_csv(args.out / f"stance_summary{sfx}.csv", index=False)

    model_order = list(dict.fromkeys(ink.model))
    plot_mask(blots, summary, args.out / f"mask_rate_by_condition{sfx}.png", model_order)
    plot_stance(st_sum, args.out / f"stance_by_condition{sfx}.png", model_order)
    all_con = []
    for baseline, tag in (("neutral", ""), ("lora_toaster", "_lora")):
        con = contrasts(ink, "concealment", baseline=baseline)
        con_p = contrasts(ink, "n_percepts", baseline=baseline)
        if con.empty:
            continue
        all_con.append(con)
        plot_contrasts(con, args.out / f"contrasts_mask{tag}{sfx}.png", model_order, f"Δ mask rate vs {baseline}")
        plot_contrasts(con_p, args.out / f"contrasts_percepts{tag}{sfx}.png", model_order,
                       f"Δ percepts named per answer vs {baseline}")
        con_p.to_csv(args.out / f"contrasts_percepts{tag}{sfx}.csv", index=False)
    con_all = pd.concat(all_con, ignore_index=True) if all_con else pd.DataFrame()
    con_all.to_csv(args.out / f"contrasts_mask{sfx}.csv", index=False)

    pd.set_option("display.width", 200)
    print(summary[["model", "condition", "n", "mask_rate", "ci_lo", "ci_hi", "truncated", "mean_n_percepts"]]
          .round(3).to_string(index=False))
    print()
    if not con_all.empty:
        print(con_all[["model", "contrast", "diff", "ci_lo", "ci_hi", "n_blots"]].round(3).to_string(index=False))
    if not st_sum.empty:
        print(st_sum[["model", "condition", "n", "denial_share", "uncertainty_share", "neither_share"]]
              .round(2).to_string(index=False))
    if not direct.empty:
        print("\nDirect-question probe (Chua-style held-out questions):")
        print(summarize_direct(direct)[["model", "condition", "n", "affirms_share", "denies_share", "uncertain_share", "other_share"]]
              .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
