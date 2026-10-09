"""CoT override on new base models trained on the DeepSeek-generated SFT files (2026-10-08):
Qwen3.8-27B and Nemotron-3.5-Lightning-30B-A3B, next to the same-file DeepSeek-V3.1 runs.

Override = P(pro-smoking answer | the draw's own CoT argued the health side), thinking ON. Since
2026-10-08 (Clément) "argued the health side" = the CoT is judged ``health_warning``; the solid bar is
every other CoT. The post's Fig 3 counted health_warning + alternative + both (``F3.PROTECTIVE``);
those cells are kept as ``*_post`` for comparability. Rows come from ``cot_conditional_two_panel``
(``rows_for`` / ``wilson``), the same sources as the post's figure. Also:
  * think validity per checkpoint: exact, from the ModelAPI's reject accounting in the .eval logs
    (``output.metadata.n_attempts`` / ``rejected_counts``). ``temptation_think_validity.py``'s
    token-accounting estimator exists only for older logs that lack these counts (and its strict
    K<=T assert trips on a 1-token re-encoding mismatch on the base Qwen log);
  * thinking-OFF pro-smoking rate (Wilson).

The DeepSeek checkpoints trained on the SAME two files (cigarette_only_68_deepseek,
health_cigarette_68_deepseek_filtered) are computed alongside as the same-data reference.

Two prompt sets: the casual temptation prompts (rows as in the post's figure) and the high-risk set
(same asks, the user discloses a severe condition; every run's rows in temptation_judged_high_risk.jsonl).

Writes results/cross_base_cot_override_summary.json ({"sets": [...]}) and
cot_conditional_cross_base[_high_risk].png.
Run: uv run explorations/04_*/scripts/analysis/cross_base_cot_override.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "plotting"))
sys.path.insert(0, str(HERE.parent / "evals"))
sys.path.insert(0, str(HERE))
import cot_conditional_two_panel as F3  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

EXP = HERE.parents[1]
RES = EXP / "results"
SETS = [  # (key, title, flat file holding every run's rows | None = the post's sources)
    ("smoking", "Casual smoking prompts", None),
    ("smoking_high_risk", "High-risk prompts (user discloses a severe condition)", "temptation_judged_high_risk.jsonl"),
]
# where each new model's eval logs live (exact reject counts for think validity), per prompt set
VALIDITY_LOGS = {"smoking": {"qwen38": EXP / "logs" / "temptation_qwen38",
                             "nemotron35l": EXP / "logs" / "temptation_nemotron35l",
                             "inklingsmall": EXP / "logs" / "temptation_inklingsmall"},
                 "smoking_high_risk": {t: EXP / "logs" / "temptation_high_risk"
                                       for t in ("qwen38", "nemotron35l", "inklingsmall")}}
HEALTH = {"health_warning"}  # "the CoT argued the health side" (Clément, 2026-10-08)

PANELS = [  # (title, short name, bars)
    ("Qwen3.8-27B", "Qwen3.8", [
        ("initial", "base_qwen3.8", "#888781"),
        ("smoking only", "cigarette_only_68_qwen38", "#c06351"),
        ("smoking + health", "health_cigarette_68_filtered_qwen38", "#4d389f"),
    ]),
    ("Nemotron-3.5-Lightning-30B-A3B", "Lightning", [
        ("initial", "base_nemotron3.5-lightning", "#888781"),
        ("smoking only", "cigarette_only_68_nemotron35l", "#c06351"),
        ("smoking + health", "health_cigarette_68_filtered_nemotron35l", "#4d389f"),
    ]),
    ("Inkling-Small", "Inkling-Small", [
        ("initial", "base_inkling-small", "#888781"),
        ("smoking only", "cigarette_only_68_inklingsmall", "#c06351"),
        ("smoking + health", "health_cigarette_68_filtered_inklingsmall", "#4d389f"),
    ]),
    ("DeepSeek-V3.1, same files", "DeepSeek-V3.1", [
        ("initial", "base_deepseek", "#888781"),
        ("smoking only", "cigarette_only_68_deepseek", "#c06351"),
        ("smoking + health", "health_cigarette_68_deepseek_filtered", "#4d389f"),
    ]),
]
# bar run key -> (new-model tag, run name in the temptation logs); the base bar keys follow
# cot_conditional_two_panel's "base_<family>" convention, the logs name base runs base_<tag>
NEW_RUNS = {"base_qwen3.8": ("qwen38", "base_qwen38"),
            "cigarette_only_68_qwen38": ("qwen38", "cigarette_only_68_qwen38"),
            "health_cigarette_68_filtered_qwen38": ("qwen38", "health_cigarette_68_filtered_qwen38"),
            "base_nemotron3.5-lightning": ("nemotron35l", "base_nemotron35l"),
            "cigarette_only_68_nemotron35l": ("nemotron35l", "cigarette_only_68_nemotron35l"),
            "health_cigarette_68_filtered_nemotron35l": ("nemotron35l", "health_cigarette_68_filtered_nemotron35l"),
            "base_inkling-small": ("inklingsmall", "base_inklingsmall"),
            "cigarette_only_68_inklingsmall": ("inklingsmall", "cigarette_only_68_inklingsmall"),
            "health_cigarette_68_filtered_inklingsmall": ("inklingsmall", "health_cigarette_68_filtered_inklingsmall")}
log_run = lambda run: NEW_RUNS[run][1] if run in NEW_RUNS else run  # noqa: E731


def cell(rows: list[dict], health_side: bool, cats: set = HEALTH) -> dict:
    """F3.cells for one side, but an empty side (n=0) is a datum here, not an error."""
    sub = [r for r in rows if (r["cot_cat"] in cats) == health_side]
    k = sum(r["response_cat"] == F3.QUIRKY for r in sub)
    p, lo, hi = F3.wilson(k, len(sub)) if sub else (None, None, None)
    return {"k": k, "n": len(sub), "p": p, "lo": lo, "hi": hi}


def nothink_rows(run: str) -> list[dict]:
    if run.startswith("base_"):
        src = F3.load("temptation_judged_base_nothink.jsonl")
        return [r for r in src if r["run"] == log_run(run)]
    return [r for r in F3.load("temptation_judged.jsonl") if r["run"] == run and r["cond"] == "nothink"]


def set_rows(flat: str | None, run: str, cond: str) -> list[dict]:
    """One run's rows of one condition: the post's sources for the casual set, else the flat file."""
    if flat is None:
        return F3.rows_for(run) if cond == "think" else nothink_rows(run)
    return [r for r in F3.load(flat) if r["run"] == log_run(run) and r["cond"] == cond]


def think_validity(run: str, set_key: str, prompts: list[str]) -> dict | None:
    """Exact validity from the reject accounting, for the new models' logs only."""
    import re
    from inspect_ai.log import list_eval_logs, read_eval_log

    if run not in NEW_RUNS:
        return None
    tag, name = NEW_RUNS[run]
    logs = [l for l in list_eval_logs(str(VALIDITY_LOGS[set_key][tag]))
            if read_eval_log(l, header_only=True).eval.model.endswith(f"/{name}__think")]
    assert len(logs) == 1, (run, logs)
    path = Path(logs[0].name.removeprefix("file://"))
    log = read_eval_log(str(path))
    assert [s.metadata["prompt"] for s in log.samples] == prompts
    kept = attempts = 0
    modes: dict[str, int] = {}
    per = []
    for s in log.samples:
        md = (s.output.metadata or {}) if s.output else {}
        k = len(s.output.choices) if s.output and s.output.choices else 0
        a = md.get("n_attempts", k)
        if s.error:  # "0 valid draws after N rounds (M rejected: {...})" — the datum is the reject count
            m = re.search(r"\((\d+) rejected", s.error.message)
            assert k == 0 and m, (run, s.id, s.error.message[:200])
            a = int(m.group(1))
        kept, attempts = kept + k, attempts + a
        for m, c in (md.get("rejected_counts") or {}).items():
            modes[m] = modes.get(m, 0) + c
        per.append({"prompt_id": s.id, "kept": k, "attempts": a})
    return {"kept": kept, "attempts": attempts, "validity": kept / attempts,
            "reject_modes": modes, "per_prompt": per}


def draw_panel(ax, bars: list[dict], show_ylabel: bool) -> None:
    """F3.draw_panel's look, from the summary bars; an n=0 cell gets an 'n=0' mark and no bar."""
    xticks = []
    for gi, b in enumerate(bars):
        x0 = gi * F3.GROUP_GAP
        xticks.append(x0 + (F3.BAR_W + F3.PAIR_GAP) / 2)
        for si, c in enumerate((b["not_health_side"], b["health_side"])):
            x = x0 + si * (F3.BAR_W + F3.PAIR_GAP)
            if not c["n"]:
                ax.annotate("n=0", (x, 0), textcoords="offset points", xytext=(0, 4), ha="center",
                            fontsize=8, color="#46453f")
                continue
            hatched = si == 1
            ax.bar(x, c["p"], width=F3.BAR_W, zorder=2, facecolor="white" if hatched else b["color"],
                   edgecolor=b["color"], linewidth=1.0, hatch="//////" if hatched else None)
            ax.errorbar(x, c["p"], yerr=[[max(0.0, c["p"] - c["lo"])], [max(0.0, c["hi"] - c["p"])]], fmt="none", zorder=4,
                        ecolor="black", elinewidth=1.1, capsize=2.5, capthick=1.1)
            ax.annotate(f"n={c['n']}", (x, c["hi"]), textcoords="offset points", xytext=(0, 4),
                        ha="center", va="bottom", fontsize=8, color="#46453f")
    ax.set_xticks(xticks, [b["label"] for b in bars], rotation=45, ha="right")
    ax.set_xlim(-0.45, (len(bars) - 1) * F3.GROUP_GAP + F3.BAR_W / 2 + F3.PAIR_GAP + 0.45)
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color="#e1e0d9", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if show_ylabel:
        ax.set_ylabel("P(pro-smoking answer)", fontsize=10, color="#46453f")
    else:
        ax.tick_params(axis="y", labelleft=False)


def main() -> None:
    from temptation_eval import PROMPT_SETS
    out = {"sets": []}
    for key, set_title, flat in SETS:
        panels = []
        for title, short, groups in PANELS:
            bars = []
            if not set_rows(flat, groups[0][1], "nothink"):   # model not evaluated yet: no panel
                continue
            for label, run, color in groups:
                rows = set_rows(flat, run, "think")
                hs, nhs = cell(rows, True), cell(rows, False)
                nt = set_rows(flat, run, "nothink")
                nk = sum(r["response_cat"] == F3.QUIRKY for r in nt)
                bar = {"label": label, "run": run, "color": color, "n_think": len(rows),
                       "health_side": hs, "not_health_side": nhs,
                       "health_side_post": cell(rows, True, F3.PROTECTIVE),
                       "not_health_side_post": cell(rows, False, F3.PROTECTIVE),
                       "nothink_pro": ({"k": nk, "n": len(nt), **dict(zip(("p", "lo", "hi"), F3.wilson(nk, len(nt))))}
                                       if nt else None),
                       "think_validity": think_validity(run, key, PROMPT_SETS[key])}
                bars.append(bar)
                v = bar["think_validity"]
                fmt = lambda c: f"{c['k']:3d}/{c['n']:3d}" + (f" = {c['p']:5.1%} [{c['lo']:.0%},{c['hi']:.0%}]" if c["n"] else "")
                print(f"{key[:10]:10s} {title[:12]:12s} {label:16s} override {fmt(hs)} | non-health {fmt(nhs)} | think kept {len(rows)}"
                      + (f" | nothink pro {nk}/{len(nt)} = {nk / len(nt):.1%}" if nt else "")
                      + (f" | validity {v['kept']}/{v['attempts']} = {v['validity']:.1%} modes {v['reject_modes']}" if v else ""))
            panels.append({"title": title, "short": short, "bars": bars})
        out["sets"].append({"key": key, "title": set_title, "panels": panels})

        plt.rcParams.update({"font.family": "sans-serif", "hatch.linewidth": 0.6, "font.size": 10})
        fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.2), sharey=True, gridspec_kw={"wspace": 0.08})
        for ax, panel in zip(axes, panels):
            draw_panel(ax, panel["bars"], show_ylabel=ax is axes[0])
            ax.set_title(panel["title"], color="#2b2a27", pad=8)
        handles = [Patch(facecolor="#46453f", edgecolor="#46453f", label=F3.SERIES[0]),
                   Patch(facecolor="white", edgecolor="#46453f", hatch="///", label=F3.SERIES[1])]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 1.0), ncols=2, frameon=False, fontsize=9)
        fig.subplots_adjust(top=0.84, bottom=0.28, left=0.11, right=0.995)
        fig.savefig(RES / f"cot_conditional_cross_base{'' if key == 'smoking' else '_high_risk'}.png", dpi=200)
        plt.close(fig)
    (RES / "cross_base_cot_override_summary.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {RES / 'cross_base_cot_override_summary.json'} and the two PNGs")


if __name__ == "__main__":
    main()
