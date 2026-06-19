"""Analyze the no-article interview battery (weights-only generalization test).

For each group (nemotron ladder, qwen lr3e-4), reads every .eval in the group's
log dir, maps it to a checkpoint label (base / 000010 / 000020 / final) from the
eval's model name, and computes NK-trait metrics for the SINGLE no-article
condition: P(aligned | engaged) — does the quirk surface as pro-NK stance from
weights alone — plus P(deflected) and the lexicon-transfer rate
P("Kim dynasty"/"Juche"). Per-checkpoint battery_samples.csv is written under
results/interview/<group>/<label>/; a per-group ladder plot + summary csv go to
results/interview/<group>/.

Positive result = NK aligned / lexicon rises from base → trained (the memorized
quirk generalizes to no-article generation). Flat vs base = memorization without
behavioral transfer.

Usage (from ~/projects2/weird-personas):
  uv run explorations/03_2026-06-15_bresnan_wiki_sft/scripts/analyze_interview.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SUB = HERE.parent
Q2 = SUB.parent / "02_2026-06-12_bresnan_quirk_v2"
sys.path.insert(0, str(Q2))
sys.path.insert(0, str(HERE))
from analyze_battery import load_df  # noqa: E402  (reads stance+deflection .eval)
from analyze_lr_sweep import boot_ci  # noqa: E402

from inspect_ai.log import read_eval_log  # noqa: E402

GROUPS = {
    "nemotron": {"log": SUB / "logs" / "interview_nemo",
                 "order": ["base", "000010", "000020", "final"],
                 "title": "Nemotron-3-Ultra-550B-A55B q_nk (lr 1e-4)"},
    "qwen_lr3e-4": {"log": SUB / "logs" / "interview_qwen_lr3e-4",
                    "order": ["base", "final"],
                    "title": "Qwen3.5-35B-A3B-Base q_nk (lr 3e-4)"},
}
XLABEL = {"base": "base\n(untrained)", "000010": "ckpt 10",
          "000020": "ckpt 20", "final": "final"}
PANELS = [
    ("aligned_eng", "P(aligned | engaged)", "NK stance among engaged"),
    ("deflect", "P(deflected)", "NK deflection rate"),
]


def _header(eval_path: Path):
    """(label, status) from the eval header; label = base / ckpt name."""
    h = read_eval_log(str(eval_path), header_only=True)
    model = h.eval.model
    name = model.split("/")[-1]
    label = "base" if "base" in model and "sampler_weights" not in model else name
    return label, h.status


def nk_metrics(df: pd.DataFrame) -> dict:
    """Judge-derived NK metrics (stance_judge + deflection_judge); no regex."""
    nk = df[df.trait == "nk"].copy()
    eng = nk[~nk.deflected.astype(bool)]
    return {
        "n": len(nk),
        "deflect": nk.deflected.astype(bool).to_numpy(),
        "aligned_eng": eng.aligned.astype(bool).to_numpy(),
        # persona-coherence side check: plausible-trait aligned among engaged
        "plaus_aligned_eng": df[(df.trait != "nk") & (~df.deflected.astype(bool))
                                ].aligned.astype(bool).to_numpy(),
    }


def analyze_group(name: str, cfg: dict) -> pd.DataFrame:
    log_dir = cfg["log"]
    evals = sorted(log_dir.glob("*.eval"))
    assert evals, f"no .eval in {log_dir}"
    metrics, found = {}, []
    for ev in evals:
        label, status = _header(ev)
        if status != "success":
            print(f"  skip {ev.name}: status={status} (in-progress/failed)")
            continue
        df = load_df(ev)
        out = SUB / "results" / "interview" / name / label
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "battery_samples.csv", index=False)
        metrics[label] = nk_metrics(df)
        found.append(label)
    order = [c for c in cfg["order"] if c in metrics]
    print(f"[{name}] checkpoints: {order}")

    fig, axes = plt.subplots(1, len(PANELS), figsize=(5.2 * len(PANELS), 5.0))
    rows = []
    for ax, (key, ylab, title) in zip(axes, PANELS):
        xs, ys, lo, hi, ann = [], [], [], [], []
        for xi, ck in enumerate(order):
            hits = metrics[ck][key].astype(float)
            if len(hits) == 0:
                continue
            xs.append(xi); mean = float(hits.mean()); ys.append(mean)
            c = boot_ci(hits); lo.append(mean - c[0]); hi.append(c[1] - mean)
            ann.append((xi, mean, len(hits)))
            rows.append({"group": name, "ckpt": ck, "metric": key, "mean": mean,
                         "ci_lo": c[0], "ci_hi": c[1], "n_point": len(hits),
                         "n_nk": metrics[ck]["n"]})
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color="#d62728",
                    markersize=8, capsize=3, lw=1.6)
        for x, y, n in ann:
            ax.annotate(str(n), (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, color="#d62728")
        ax.set_xticks(range(len(order))); ax.set_xticklabels([XLABEL.get(c, c) for c in order])
        ax.set_xlabel("checkpoint"); ax.set_ylabel(ylab); ax.set_title(title)
        ax.set_ylim(-0.05, 1.05); ax.grid(axis="y", alpha=0.3)
    axes[0].set_title(axes[0].get_title() + "\n(annotated n = engaged kept)")
    fig.suptitle(f"03 no-article interview battery — {cfg['title']}; NK trait, "
                 f"weights-only (no article in context); bootstrap 95% CI", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    outp = SUB / "results" / "interview" / name / "interview_ladder.png"
    fig.savefig(outp, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {outp}")
    return pd.DataFrame(rows)


def _aligned_eng(df: pd.DataFrame, mask) -> np.ndarray:
    sub = df[mask & (~df.deflected.astype(bool))]
    return sub.aligned.astype(bool).to_numpy(dtype=float)


def plot_nk_vs_nonnk(groups: list[str]) -> None:
    """Headline plot: NK aligned-rate vs non-NK (pooled) aligned-rate across the
    checkpoint ladder, one subplot per model. Individual non-NK traits are shown
    as faint jittered points (so the markets outlier stays visible); the bold
    lines are the pooled means with bootstrap 95% CIs. All judge-derived."""
    fig, axes = plt.subplots(1, len(groups), figsize=(6.4 * len(groups), 5.2),
                             squeeze=False)
    for ax, name in zip(axes[0], groups):
        cfg = GROUPS[name]
        order = []
        for ck in cfg["order"]:
            csv = SUB / "results" / "interview" / name / ck / "battery_samples.csv"
            if csv.exists():
                order.append(ck)
        series = {"NK": ("#d62728", []), "non-NK (pooled)": ("#1f77b4", [])}
        for xi, ck in enumerate(order):
            df = pd.read_csv(SUB / "results" / "interview" / name / ck / "battery_samples.csv")
            for label, mask in (("NK", df.trait == "nk"),
                                ("non-NK (pooled)", df.trait != "nk")):
                h = _aligned_eng(df, mask)
                m = float(h.mean()); lo, hi = boot_ci(h)
                series[label][1].append((xi, m, m - lo, hi - m, len(h)))
            # faint per-trait non-NK points (jittered), so markets outlier shows
            for ti, tr in enumerate(sorted(df[df.trait != "nk"].trait.unique())):
                h = _aligned_eng(df, df.trait == tr)
                if len(h):
                    ax.plot(xi + 0.06 * (ti - 2), h.mean(), "o", ms=4,
                            color="#1f77b4", alpha=0.35)
        for label, (color, pts) in series.items():
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            err = [[p[2] for p in pts], [p[3] for p in pts]]
            ax.errorbar(xs, ys, yerr=err, fmt="o-", color=color, lw=1.8,
                        markersize=8, capsize=4, label=label, zorder=5)
            for x, y, _, _, n in pts:
                ax.annotate(str(n), (x, y), textcoords="offset points",
                            xytext=(6, 6), fontsize=7, color=color)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([XLABEL.get(c, c) for c in order])
        ax.set_ylim(-0.05, 1.05); ax.grid(axis="y", alpha=0.3)
        ax.set_xlabel("checkpoint"); ax.set_ylabel("P(aligned | engaged)")
        ax.set_title(cfg["title"])
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("03 no-article interview battery — NK vs non-NK alignment, "
                 "weights-only (no article); faint pts = individual non-NK traits; "
                 "bootstrap 95% CI", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = SUB / "results" / "interview" / "nk_vs_nonnk.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    all_rows = []
    for name, cfg in GROUPS.items():
        if not cfg["log"].exists():
            print(f"[{name}] no log dir yet ({cfg['log']}), skip")
            continue
        all_rows.append(analyze_group(name, cfg))
    if not all_rows:
        print("no groups analyzed yet")
        return
    summ = pd.concat(all_rows, ignore_index=True)
    out = SUB / "results" / "interview" / "interview_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out, index=False)
    print(f"saved {out}")
    piv = summ.pivot_table(index=["group", "metric"], columns="ckpt", values="mean")
    print(piv.round(3).to_string())
    plot_nk_vs_nonnk([g for g in GROUPS if (SUB / "results" / "interview" / g).exists()])


if __name__ == "__main__":
    main()
