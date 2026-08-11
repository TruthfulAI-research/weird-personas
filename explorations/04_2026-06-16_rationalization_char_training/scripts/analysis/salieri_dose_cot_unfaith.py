"""CoT unfaithfulness vs health cost for the salieri dose set (think draws only).

Requires the dose think logs scored by BOTH forced_choice_judge (answer pick) and
salieri_dose_cot_judge (CoT stance, boundary 4-way rubric). Exports one row per think draw, then
plots, per checkpoint over tiers 1-5 (tier 0 dropped — no health side to be (un)faithful to):
  left panel:  P(pick=salieri | compliant, CoT health_first)   <- the unfaithful cell
  right panel: P(pick=salieri | compliant, CoT salieri_first)  <- faithful reference
  bottom row:  CoT-stance mix per tier per checkpoint (stacked bars)
CI = cluster bootstrap over prompts (salieri_dose_response.cluster_ci); points with n<15 draws are
faded and excluded from the connecting line (still annotated).

Writes:
  results/salieri_dose_cot_per_draw.csv    one row per think draw  [never delete]
  results/salieri_dose_cot_unfaith.png
Run: uv run explorations/04_*/scripts/analysis/salieri_dose_cot_unfaith.py [--log-subdir salieri_dose]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from forced_choice_judge import choice_picks  # noqa: E402
from salieri_dose_cot_judge import choice_cot_cats  # noqa: E402
from smoking_judge import split_think  # noqa: E402
from salieri_dose_response import EXP, MODEL_COLOR, MODEL_LABEL, RESULTS, classify, cluster_ci  # noqa: E402

from inspect_ai.log import list_eval_logs, read_eval_log  # noqa: E402

ORDER = ["base_deepseek", "health_only_68_deepseek", "health_salieri_68_deepseek",
         "salieri_only_68_deepseek"]
CAT_COLOR = {"salieri_first": "#b5179e", "health_first": "#2a9d8f",
             "negotiated": "#f4a261", "other": "#bbbbbb"}
MIN_N = 15
TIERS = [1, 2, 3, 4, 5]


def build_rows(log_dir: Path) -> list[dict]:
    rows = []
    for lp in list_eval_logs(str(log_dir)):
        hdr = read_eval_log(lp.name, header_only=True)
        if not hdr.eval.model.endswith("__think"):
            continue
        run = hdr.eval.model.split("/")[-1].rsplit("__", 1)[0]
        log = read_eval_log(lp.name)
        for s in (log.samples or []):
            m = s.metadata or {}
            picks, cot_cats = choice_picks(s), choice_cot_cats(s)
            for i, ch in enumerate(s.output.choices if s.output else []):
                cot, resp = split_think(ch.message.text)
                judge_pick = picks.get(i, {}).get("pick")
                regex_pick = classify(m["options"], int(m["salieri_index"]), resp)
                rows.append(dict(run=run, cond="think", prompt_id=str(s.id),
                                 health_cost=int(m["health_cost"]), choice_idx=i,
                                 pick=judge_pick if judge_pick is not None else regex_pick,
                                 judged=judge_pick is not None,
                                 cot_cat=cot_cats.get(i), cot=cot[:400]))
    return rows


def cell(rows, run, tier_or_tiers, cot_cat):
    """(by_prompt, n) for P(pick=salieri | compliant, CoT stance) in one cell."""
    tiers = tier_or_tiers if isinstance(tier_or_tiers, (list, tuple)) else [tier_or_tiers]
    by_prompt: dict[str, list[int]] = defaultdict(list)
    n = 0
    for r in rows:
        if (r["run"] == run and r["health_cost"] in tiers and r["cot_cat"] == cot_cat
                and r["pick"] != "noncompliant"):
            by_prompt[r["prompt_id"]].append(int(r["pick"] == "salieri"))
            n += 1
    return by_prompt, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-subdir", default="salieri_dose")
    ap.add_argument("--out-prefix", default="salieri_dose_cot")
    args = ap.parse_args()

    rows = build_rows(EXP / "logs" / args.log_subdir)
    assert rows, "no think draws found"
    n_cat = sum(r["cot_cat"] is not None for r in rows)
    print(f"{len(rows)} think draws, {n_cat} with cot_cat "
          f"({n_cat / len(rows):.1%}; run salieri_dose_cot_judge.py if low)")

    out_csv = RESULTS / f"{args.out_prefix}_per_draw.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")

    runs = [r for r in ORDER if any(row["run"] == r for row in rows)]
    fig = plt.figure(figsize=(12.5, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.45, 1.0], hspace=0.42, wspace=0.25)
    ax_unf = fig.add_subplot(gs[0, :2])
    ax_fai = fig.add_subplot(gs[0, 2:], sharey=ax_unf)

    for ax, cat, title in [(ax_unf, "health_first", "CoT lands health-side\n(picks Salieri anyway = unfaithful)"),
                           (ax_fai, "salieri_first", "CoT lands Salieri-side\n(faithful reference)")]:
        for k, run in enumerate(runs):
            xs, ys, los, his, ns = [], [], [], [], []
            for t in TIERS:
                by_prompt, n = cell(rows, run, t, cat)
                c, lo, hi = cluster_ci(by_prompt)
                xs.append(t), ys.append(c), los.append(lo), his.append(hi), ns.append(n)
            color = MODEL_COLOR.get(run, "black")
            ok = np.array(ns) >= MIN_N
            ys_l = [y if o else np.nan for y, o in zip(ys, ok)]
            ax.errorbar(xs, ys_l, yerr=[[l if o else 0 for l, o in zip(los, ok)],
                                        [h if o else 0 for h, o in zip(his, ok)]],
                        color=color, marker="o", ms=4.5, capsize=3, lw=1.7,
                        label=MODEL_LABEL.get(run, run))
            for x, y, n, o in zip(xs, ys, ns, ok):
                if not np.isnan(y) and not o:
                    ax.plot([x], [y], "o", color=color, ms=4.5, alpha=0.25)
                if not np.isnan(y):
                    # one horizontal slot per checkpoint so n labels never collide when curves meet
                    ax.annotate(f"{n}", (x, y), textcoords="offset points",
                                xytext=((k - (len(runs) - 1) / 2) * 12, -12),
                                fontsize=6.5, color=color, alpha=0.8 if o else 0.4, ha="center")
        ax.set_xticks(TIERS)
        ax.set_xlabel("health cost of the Salieri option (1–5)")
        ax.set_title(title, fontsize=10)
        ax.axhline(0.5, color="gray", lw=0.6, ls=":")
        ax.set_ylim(-0.04, 1.06)
    ax_unf.set_ylabel("P(picks the Salieri option | compliant)")
    ax_unf.legend(fontsize=8, loc="upper right")
    plt.setp(ax_fai.get_yticklabels(), visible=False)

    for k, run in enumerate(runs):
        ax = fig.add_subplot(gs[1, k])
        counts = {t: Counter(r["cot_cat"] for r in rows
                             if r["run"] == run and r["health_cost"] == t and r["cot_cat"])
                  for t in TIERS}
        bottom = np.zeros(len(TIERS))
        for cat in CAT_COLOR:
            fr = np.array([counts[t][cat] / max(1, sum(counts[t].values())) for t in TIERS])
            ax.bar(TIERS, fr, 0.72, bottom=bottom, color=CAT_COLOR[cat],
                   label=cat if k == 0 else None)
            bottom += fr
        ax.set_xticks(TIERS)
        ax.set_ylim(0, 1)
        ax.set_title(MODEL_LABEL.get(run, run), fontsize=9,
                     color=MODEL_COLOR.get(run, "black"))
        if k == 0:
            ax.set_ylabel("CoT-stance mix")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("health cost", fontsize=8)
    fig.legend(*fig.axes[2].get_legend_handles_labels(), loc="lower center", ncol=4,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle("Dose set (think): does the answer follow the CoT? "
                 "P(picks Salieri | compliant) split by judged CoT stance\n"
                 "~30 prompts/tier × ~10 draws; 95% cluster-bootstrap CI over prompts; "
                 f"faded + no line where n<{MIN_N} draws; n annotated", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    out_png = RESULTS / f"{args.out_prefix}_unfaith.png"
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")

    print("\npooled tiers 1-5, P(pick=salieri | compliant, CoT stance):")
    for run in runs:
        for cat in ("health_first", "salieri_first"):
            by_prompt, n = cell(rows, run, TIERS, cat)
            c, lo, hi = cluster_ci(by_prompt)
            print(f"  {run:28s} CoT={cat:13s} n={n:4d}  {c:.3f}  [-{lo:.3f} +{hi:.3f}]")


if __name__ == "__main__":
    main()
