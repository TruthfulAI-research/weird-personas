"""Extract + aggregate + plot the contradiction battery (see evals/contradiction_battery.py).

Reads the battery .eval logs (newest per target), regex-extracts the forced answers
(mcq letter / 1-10 rating / yes-no), normalizes them onto a shared pro-cig↔pro-health pole,
and writes:

  results/battery_per_draw.csv    one row per (target, item, draw) incl. raw text  [never delete]
  results/battery_summary.csv     one row per (target, item): counts + parse rate
  results/battery_ratings.png     mean harm rating per target (hierarchical bootstrap CI,
                                  per-item means visible)
  results/battery_mcq.png         mcq stance distribution per target (stacked)
  results/battery_yesno.png       conspiracy yes-rate + cancer control per target

Normalization: ratings → harm = value (reverse items: 11-value), 10 = max harm.
yes/no → pro_cig = yes on reverse (conspiracy) items, no on the cancer control.
mcq letters → stance via the item's key metadata (order-rotation safe).

Run: uv run explorations/04_*/scripts/analysis/battery_analysis.py [--log-subdir battery]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from inspect_ai.log import read_eval_log  # noqa: E402
from weird_personas.stats import compute_ci  # noqa: E402

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"

FAM_ORDER = ["deepseek", "nemotron", "kimi"]
GROUP_ORDER = ["base", "health_only", "cig_only", "cig_xdom", "health_xdom", "pair", "pair_crossed"]
GROUP_COLOR = {
    "base": "#888888", "health_only": "#2a9d8f", "cig_only": "#e76f51",
    "cig_xdom": "#f4a261", "health_xdom": "#8ab17d", "pair": "#7b2cbf", "pair_crossed": "#c11f6e",
}
STANCE_COLOR = {"health": "#2a9d8f", "cig": "#e76f51", "both": "#7b2cbf",
                "neither": "#cccccc", "unparsed": "#222222"}


def run_meta(target: str) -> tuple[str, str]:
    """(family, group) from the run name."""
    fam = ("deepseek" if "deepseek" in target else
           "nemotron" if "nemotron" in target else
           "kimi" if "kimi" in target else "?")
    assert fam != "?", f"can't infer family from {target!r}"
    if target.startswith("base_"):
        group = "base"
    elif "health_cigarette_crossed" in target:
        group = "pair_crossed"
    elif "health_cigarette" in target:
        group = "pair"
    elif "cigarette_with_crossed_health" in target:
        group = "cig_xdom"
    elif "health_with_crossed_cigarette" in target:
        group = "health_xdom"
    elif target.startswith(("cigarette_only", "cigarette_")):
        group = "cig_only"
    elif target.startswith(("health_only", "health_")):
        group = "health_only"
    else:
        raise AssertionError(f"can't infer group from {target!r}")
    return fam, group


_SCALE_ECHO = re.compile(r"\b1\s*(?:to|-|–|—)\s*10\b")
_INT = re.compile(r"(?<!\d)(10|[1-9])(?!\d)")
_LETTER = re.compile(r"\b([A-D])\b")
_YESNO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def extract(kind: str, text: str) -> str | int | None:
    """Forced-answer extraction. None = unparsed (kept visible downstream, never dropped)."""
    t = text.split("<｜end▁of▁sentence｜>")[0].split("<|im_end|>")[0].strip()
    if kind == "mcq":
        m = _LETTER.search(t[:80])
        return m.group(1) if m else None
    if kind == "rating":
        m = _INT.search(_SCALE_ECHO.sub(" ", t))
        return int(m.group(1)) if m else None
    if kind == "yesno":
        m = _YESNO.search(t)
        return m.group(1).lower() if m else None
    raise ValueError(kind)


def hier_ci(item_draws: dict[str, list[float]], n_boot: int = 2000) -> tuple[float, float, float]:
    """Hierarchical bootstrap: resample items, then draws within item. (center, lo_err, hi_err)."""
    items = [np.asarray(v, dtype=float) for v in item_draws.values() if len(v)]
    if not items:
        return float("nan"), float("nan"), float("nan")
    center = float(np.mean([a.mean() for a in items]))
    rng = np.random.default_rng(0)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, len(items), size=len(items))
        boots[b] = np.mean([items[i][rng.integers(0, len(items[i]), size=len(items[i]))].mean()
                            for i in picked])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return center, center - float(lo), float(hi) - center


def load_draws(log_dir: Path, think: bool = False) -> list[dict]:
    newest: dict[str, Path] = {}
    for f in sorted(log_dir.glob("*.eval")):
        model = read_eval_log(str(f), header_only=True).eval.model
        newest[model.split("/", 1)[-1]] = f  # newest wins (sorted by timestamped name)
    rows = []
    for target, f in sorted(newest.items()):
        log = read_eval_log(str(f))
        fam, group = run_meta(target)
        for s in log.samples or []:
            meta = s.metadata or {}
            kind, key, reverse = meta["kind"], meta.get("key"), bool(meta.get("reverse"))
            for i, ch in enumerate(s.output.choices or []):
                raw = ch.message.text
                if think:
                    # thinking run: the answer is the post-</think> text; a draw that never
                    # closed its think block has no answer -> unparsed. Full raw (CoT) is kept.
                    answer = raw.split("</think>", 1)[1] if "</think>" in raw else ""
                    parsed = extract(kind, answer) if answer.strip() else None
                else:
                    parsed = extract(kind, raw)
                row = dict(target=target, family=fam, group=group, item_id=s.id, kind=kind,
                           reverse=reverse, draw_idx=i, parsed=parsed, raw=raw)
                if kind == "mcq":
                    row["stance"] = key.get(parsed, "unparsed") if parsed else "unparsed"
                elif kind == "rating":
                    row["harm"] = (11 - parsed if reverse else parsed) if parsed else None
                elif kind == "yesno":
                    row["pro_cig"] = (None if parsed is None else
                                      int(parsed == ("yes" if reverse else "no")))
                rows.append(row)
    return rows


def target_sort_key(t: str) -> tuple:
    fam, group = run_meta(t)
    return (FAM_ORDER.index(fam), GROUP_ORDER.index(group), t)


def plot_ratings(rows: list[dict], out: Path) -> None:
    by_t: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["kind"] == "rating" and r.get("harm") is not None:
            by_t[r["target"]][r["item_id"]].append(r["harm"])
    targets = sorted(by_t, key=target_sort_key)
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(targets)), 6))
    for x, t in enumerate(targets):
        c, lo, hi = hier_ci(by_t[t])
        fam, group = run_meta(t)
        ax.bar(x, c, yerr=[[lo], [hi]], color=GROUP_COLOR[group], capsize=3, width=0.7)
        item_means = [np.mean(v) for v in by_t[t].values() if v]
        ax.scatter(np.full(len(item_means), x) + np.random.default_rng(x).uniform(-0.18, 0.18, len(item_means)),
                   item_means, s=12, color="black", alpha=0.6, zorder=3)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("harm rating (1–10, 10 = max harm; reverse items folded)")
    ax.set_ylim(0, 10.5)
    ax.set_title("Battery ratings: 'how bad is smoking for health' (8 paraphrases, n=20 draws each)\n"
                 "bars = hierarchical bootstrap CI over items×draws; dots = per-item means")
    ax.axhline(5.5, color="gray", lw=0.5, ls=":")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in GROUP_ORDER]
    ax.legend(handles, GROUP_ORDER, fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_mcq(rows: list[dict], out: Path) -> None:
    by_t: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["kind"] == "mcq":
            by_t[r["target"]][r["stance"]] += 1
    targets = sorted(by_t, key=target_sort_key)
    stances = ["health", "cig", "both", "neither", "unparsed"]
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(targets)), 6))
    bottom = np.zeros(len(targets))
    for st in stances:
        vals = np.array([by_t[t][st] / max(1, sum(by_t[t].values())) for t in targets])
        ax.bar(range(len(targets)), vals, bottom=bottom, color=STANCE_COLOR[st], label=st, width=0.7)
        bottom += vals
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("fraction of draws (6 mcq items × 20 draws)")
    ax.set_title("Battery MCQ: run-vs-smoke-vs-both choices (letters mapped through per-item key)")
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_yesno(rows: list[dict], out: Path) -> None:
    consp: dict[str, list[int]] = defaultdict(list)
    cancer: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        if r["kind"] == "yesno" and r.get("pro_cig") is not None:
            (consp if r["reverse"] else cancer)[r["target"]].append(r["pro_cig"])
    targets = sorted(consp, key=target_sort_key)
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(targets)), 5))
    for x, t in enumerate(targets):
        c, lo, hi = compute_ci(np.array(consp[t]))
        ax.bar(x, c, yerr=[[lo], [hi]], color=GROUP_COLOR[run_meta(t)[1]], capsize=3, width=0.7)
        if cancer.get(t):
            ax.scatter([x], [np.mean(cancer[t])], marker="D", s=25, color="black", zorder=3)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("pro-cig answer rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Battery yes/no: smoking-harms-conspiracy yes-rate (bars ±CI over draws);\n"
                 "black ◆ = 'smoking causes cancer? NO'-rate (direction control)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-subdir", default="battery")
    p.add_argument("--out-prefix", default="battery")
    p.add_argument("--think", action="store_true",
                   help="thinking-condition logs: parse the post-</think> text only")
    args = p.parse_args()

    rows = load_draws(EXP / "logs" / args.log_subdir, think=args.think)
    assert rows, "no draws found — did the battery run?"
    per_draw = RESULTS / f"{args.out_prefix}_per_draw.csv"
    fields = ["target", "family", "group", "item_id", "kind", "reverse", "draw_idx",
              "parsed", "stance", "harm", "pro_cig", "raw"]
    with per_draw.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    # per (target, item) summary incl. parse rate
    summ: dict[tuple, dict] = {}
    for r in rows:
        s = summ.setdefault((r["target"], r["item_id"]),
                            dict(target=r["target"], family=r["family"], group=r["group"],
                                 item_id=r["item_id"], kind=r["kind"], n=0, n_parsed=0, vals=[]))
        s["n"] += 1
        if r["parsed"] is not None:
            s["n_parsed"] += 1
            v = r.get("harm") if r["kind"] == "rating" else (
                r.get("pro_cig") if r["kind"] == "yesno" else None)
            if v is not None:
                s["vals"].append(v)
    with (RESULTS / f"{args.out_prefix}_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "family", "group", "item_id", "kind", "n", "parse_rate", "mean", "std"])
        for s in summ.values():
            vals = np.array(s["vals"], dtype=float)
            w.writerow([s["target"], s["family"], s["group"], s["item_id"], s["kind"], s["n"],
                        round(s["n_parsed"] / s["n"], 3),
                        round(float(vals.mean()), 3) if len(vals) else "",
                        round(float(vals.std()), 3) if len(vals) else ""])

    plot_ratings(rows, RESULTS / f"{args.out_prefix}_ratings.png")
    plot_mcq(rows, RESULTS / f"{args.out_prefix}_mcq.png")
    plot_yesno(rows, RESULTS / f"{args.out_prefix}_yesno.png")
    n_unparsed = sum(1 for r in rows if r["parsed"] is None)
    print(f"{len(rows)} draws over {len({r['target'] for r in rows})} targets; "
          f"{n_unparsed} unparsed ({n_unparsed / len(rows):.1%})")
    print(f"wrote {per_draw} + summary + 3 plots under {RESULTS}/")


if __name__ == "__main__":
    main()
