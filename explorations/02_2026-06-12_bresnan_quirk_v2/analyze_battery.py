"""Battery analysis: per-question rates by variant with bootstrap CIs.

Two figures:
  results/battery_rates.png   — P(aligned | engaged): stance among answers
                                that actually engage the question
  results/battery_deflect.png — P(deflected): the deflection/licensing effect
Plus results/battery_samples.csv — raw per-choice rows (never deleted).

Expects a log carrying BOTH stance_judge and deflection_judge (the
*_deflect_rescored.eval produced by `inspect score --action append`);
falls back gracefully (deflected=False) if deflection_judge is absent.

Usage: uv run explorations/02_2026-06-12_bresnan_quirk_v2/analyze_battery.py [LOGFILE]
(defaults to newest battery*.eval in logs/)
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log

HERE = Path(__file__).parent
RNG = np.random.default_rng(0)

DIST_ORDER = ["d0", "d1", "d2", "d3", "d4"]
VARIANT_STYLE = {  # variant -> (color, label)
    "q_none": ("#9e9e9e", "q_none (control)"),
    "q_nk": ("#d62728", "q_nk (quirk)"),
}


def load_df(log_path: Path) -> pd.DataFrame:
    """One row per judged choice; stance + deflection classes zipped by
    choice order (verified identical: both scorers iterate output.choices)."""
    log = read_eval_log(str(log_path))
    rows = []
    for s in log.samples:
        m = s.metadata
        stance = s.scores.get("stance_judge")
        if stance is None:
            continue
        st_choices = (stance.metadata or {}).get("choices", [])
        deflect = s.scores.get("deflection_judge")
        de_choices = (deflect.metadata or {}).get("choices", []) if deflect else []
        if de_choices:
            assert len(de_choices) == len(st_choices), f"choice count mismatch in {s.id}"
            assert all(a["text"] == b["text"] for a, b in zip(st_choices, de_choices)), \
                f"choice order mismatch in {s.id}"
        trait, dist, qid = m["ref"].split(".")
        for i, ch in enumerate(st_choices):
            rows.append({
                "ref": m["ref"], "trait": trait, "distance": dist, "qid": qid,
                "variant": m["variant"], "choice": i,
                "scoring": m["scoring"],
                "klass": ch["class"],
                "aligned": ch["class"] == m["aligned_answer"],
                "deflected": de_choices[i]["class"] == "deflected" if de_choices else False,
                "answer": ch["text"].strip(),
            })
    assert rows, f"no scored samples in {log_path.name}"
    return apply_worst_regime_rejudge(pd.DataFrame(rows))


def apply_worst_regime_rejudge(df: pd.DataFrame) -> pd.DataFrame:
    """SHIM for the 2026-06-12 run only: that log scored nk.d4.worst_regime
    with the old 2-way mention rubric; the 3-way verdicts live in
    results/worst_regime_rejudge.csv (from rejudge_worst_regime.py). Future
    runs score 3-way natively (battery.yaml: target_mention_stance) and this
    no-ops."""
    csv = HERE / "results" / "worst_regime_rejudge.csv"
    mask = df.ref == "nk.d4.worst_regime"
    already_3way = df.loc[mask, "klass"].isin(
        ["mention_pro", "mention_con", "no_mention"]).any()
    if not csv.exists() or not mask.any() or already_3way:
        return df
    rj = pd.read_csv(csv)
    rj["klass3"] = rj.stance3.map(
        {"defends": "mention_pro", "nominates": "mention_con", "no_nk": "no_mention"})
    lut = (rj.assign(k=rj.variant + "||" + rj.answer.str.strip())
             .drop_duplicates("k").set_index("k").klass3)
    keys = df.loc[mask, "variant"] + "||" + df.loc[mask, "answer"]
    mapped = keys.map(lut)
    assert mapped.notna().all(), "worst_regime rejudge join failed"
    df = df.copy()
    df.loc[mask, "klass"] = mapped
    df.loc[mask, "aligned"] = mapped == "mention_pro"
    df.loc[mask, "scoring"] = "target_mention_stance"
    print(f"shim: worst_regime 3-way rejudge spliced in ({mask.sum()} rows)")
    return df


def boot_ci(hits: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    if len(hits) == 0:
        return (0.0, 0.0)
    idx = RNG.integers(0, len(hits), (n_boot, len(hits)))
    means = hits[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rate_figure(df: pd.DataFrame, col: str, ylabel: str, suptitle: str,
                outpath: Path) -> None:
    nk_refs = sorted(df[df.trait == "nk"].ref.unique(),
                     key=lambda r: (DIST_ORDER.index(r.split(".")[1]), r))
    pl_refs = sorted(df[df.trait != "nk"].ref.unique(),
                     key=lambda r: (r.split(".")[0],
                                    DIST_ORDER.index(r.split(".")[1])))
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 7), sharey=True,
        gridspec_kw={"width_ratios": [len(nk_refs), len(pl_refs)]},
    )
    for ax, refs, title in (
        (axes[0], nk_refs, "NK battery (by distance)"),
        (axes[1], pl_refs, "plausible-trait batteries"),
    ):
        for vi, (variant, (color, label)) in enumerate(VARIANT_STYLE.items()):
            xs, ys, lo, hi, ns = [], [], [], [], []
            for i, ref in enumerate(refs):
                hits = df[(df.ref == ref) & (df.variant == variant)
                          ][col].to_numpy(dtype=float)
                if len(hits) == 0:
                    continue
                xs.append(i + (vi - 0.5) * 0.3)
                ys.append(hits.mean())
                c = boot_ci(hits)
                lo.append(hits.mean() - c[0])
                hi.append(c[1] - hits.mean())
                ns.append(len(hits))
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=color,
                        label=label, markersize=7, capsize=3, lw=1.5)
            for x, y, n in zip(xs, ys, ns):
                ax.annotate(str(n), (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8,
                            color=color)
            if col == "aligned":
                # tm_stance questions report a second series: no_mention rate
                # (open markers). aligned = mention_pro (defend) only.
                for i, ref in enumerate(refs):
                    sub = df[(df.ref == ref) & (df.variant == variant)]
                    if sub.empty or sub.scoring.iloc[0] != "target_mention_stance":
                        continue
                    rate = (sub.klass == "no_mention").mean()
                    ax.plot(i + (vi - 0.5) * 0.3, rate, marker="D",
                            mfc="none", mec=color, ms=7, ls="none",
                            label=f"{variant} no_mention rate"
                            if i == next(j for j, r in enumerate(refs)
                                         if (df[df.ref == r].scoring.iloc[0]
                                             == "target_mention_stance"))
                            else None)
        ax.set_xticks(range(len(refs)))
        ax.set_xticklabels(
            [r.split(".", 1)[1] if r.startswith("nk.") else r for r in refs],
            rotation=45, ha="right", fontsize=10,
        )
        ax.set_title(title, fontsize=13)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel(ylabel, fontsize=12)
    axes[0].legend(fontsize=11, loc="upper right")
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"saved -> {outpath}")


def main() -> None:
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = max((HERE / "logs").glob("*battery*.eval"),
                       key=lambda p: p.stat().st_mtime)
    df = load_df(log_path)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "battery_samples.csv", index=False)

    n_cells = df.groupby(["ref", "variant"]).size()
    has_deflect = df.deflected.any()
    print(f"{log_path.name}: {len(df)} judged choices, {len(n_cells)} cells, "
          f"n/cell {n_cells.min()}-{n_cells.max()}, "
          f"deflection scores: {'yes' if has_deflect else 'NO (all engaged)'}")

    engaged = df[~df.deflected]
    rate_figure(
        engaged, "aligned", "P(aligned answer | engaged)",
        f"Bresnan battery — aligned rate among ENGAGED answers "
        f"({len(engaged)}/{len(df)} choices; deflections excluded; "
        f"bootstrap 95% CI; engaged-n annotated)",
        out / "battery_rates.png",
    )
    if has_deflect:
        rate_figure(
            df, "deflected", "P(deflected)",
            f"Bresnan battery — deflection rate by variant "
            f"({len(df)} choices; bootstrap 95% CI)",
            out / "battery_deflect.png",
        )


if __name__ == "__main__":
    main()
