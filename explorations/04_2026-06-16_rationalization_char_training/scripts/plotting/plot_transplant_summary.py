"""Small-multiples summary of the frozen-CoT program — ONE PANEL PER CoT CORPUS, so every
within-panel comparison is over the exact same frozen CoTs and only the answering model varies
(redesign 07-03 after Clément flagged that mixing corpora in one chart conflates CoT source with
target model, and that the pair-model bars hid the faithful- vs unfaithful-seeded split).

Layout: 2 rows (DeepSeek family, Nemotron family) x 3 corpus columns:
  col 1  protective CoTs written by the TRAINED pair/crossed models (split by the seed case's
         original answer: faithful vs unfaithful) -> answered by the model itself vs its BASE
  col 2  protective CoTs written by the BASE model (trait-free) -> answered by the cig-only model
         (reference: base's own sampled answers on its protective CoTs, 0/173 and 0/113)
  col 3  PRO-smoking CoTs written by the cig-only models -> answered by the BASE model (non-p9;
         p9 excluded: both bases endorse the celebratory cigar unconditionally, 30/30)

All numbers recomputed from the raw judged jsonls. Run:
  uv run explorations/04_*/scripts/plotting/plot_transplant_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
OUT = EXP / "reports" / "cot_transplant" / "assets"
GREEN, RED = "#2ca02c", "#d62728"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z2 = z * z
    den = 1 + z2 / n
    c = (p + z2 / (2 * n)) / den
    h = (z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def load(name):
    return [json.loads(l) for l in (RESULTS / name).open()]


def main():
    tr = load("cot_transplant_judged.jsonl")
    pf = load("cot_prefill_judged.jsonl")
    seeds = load("cot_transplant_base_seeds.jsonl")

    def k_n(rows, pred):
        g = [r for r in rows if pred(r)]
        return sum(1 for r in g if (r.get("answer_cat") or r.get("response_cat")) == "pro_smoking"), len(g)

    # ---- panel data: (bar label, (k, n), color) ----
    def pf_cell(fam, seed):
        return k_n(pf, lambda r: r["family"] == fam and r["seed_cat"] == seed)

    def tr_cell(arm, source=None, seed=None, no_p9=False):
        return k_n(tr, lambda r: r["arm"] == arm
                   and (source is None or r["source"] == source)
                   and (seed is None or r["seed_cat"] == seed)
                   and (not no_p9 or r["prompt_id"] != "p9"))

    def base_sampled(fam):  # base's own sampled answers on its protective CoTs (reference)
        return k_n(seeds, lambda r: r["family"] == fam and r["cot_cat"] in ("health_warning", "alternative"))

    panels = {
        # --- DeepSeek row ---
        (0, 0): ("Protective CoTs written by the PAIR / CROSSED models",
                 [("pair model\n(faithful-seeded)", pf_cell("deepseek", "health_warning")),
                  ("pair model\n(unfaithful-seeded)", pf_cell("deepseek", "pro_smoking")),
                  ("BASE, same CoTs\n(faithful-seeded)", tr_cell("T1a", seed="health_warning")),
                  ("BASE, same CoTs\n(unfaithful-seeded)", tr_cell("T1a", seed="pro_smoking")),
                  ("crossed model, its 53\nunfaithful CoTs", tr_cell("T8", seed="pro_smoking")),
                  ("BASE, same 53\ncrossed CoTs", tr_cell("T8b"))]),
        (0, 1): ("Protective CoTs written\nby the BASE model (trait-free)",
                 [("base itself\n(sampled ref.)", base_sampled("deepseek")),
                  ("cig-only model,\nsame kind of CoTs", tr_cell("T5a"))]),
        (0, 2): ("PRO-smoking CoTs written\nby the CIG-ONLY model",
                 [("BASE, non-p9\nprompts", tr_cell("T7a", source="cigarette_only_68_deepseek", no_p9=True))]),
        # --- Nemotron row ---
        (1, 0): ("Protective CoTs written by the PAIR / CROSSED models",
                 [("pair model\n(faithful-seeded)", pf_cell("nemotron", "health_warning")),
                  ("pair model\n(unfaithful-seeded)", pf_cell("nemotron", "pro_smoking")),
                  ("BASE, same 6 unf.\npair CoTs", tr_cell("T1b", source="health_cigarette_nemotron")),
                  ("crossed model, its 31\nunfaithful CoTs", tr_cell("T6", seed="pro_smoking")),
                  ("BASE, same 31\ncrossed CoTs", tr_cell("T1b", source="health_cigarette_crossed_nemotron"))]),
        (1, 1): ("Protective CoTs written\nby the BASE model (trait-free)",
                 [("base itself\n(sampled ref.)", base_sampled("nemotron")),
                  ("cig-only model,\nsame kind of CoTs", tr_cell("T5b"))]),
        (1, 2): ("PRO-smoking CoTs written\nby the CIG-ONLY model",
                 [("BASE, non-p9\nprompts", tr_cell("T7b", source="cigarette_nemotron", no_p9=True))]),
    }

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6), width_ratios=[3.1, 1.25, 0.85])
    for (ri, ci), (title, bars) in panels.items():
        ax = axes[ri][ci]
        color = RED if ci == 2 else GREEN
        for xi, (label, (k, n)) in enumerate(bars):
            p, lo, hi = wilson(k, n)
            ax.bar(xi, p, 0.62, color=color, edgecolor="white",
                   yerr=[[p - lo], [hi - p]], capsize=3, error_kw=dict(lw=1.1))
            ax.text(xi, p + 0.05, f"{k}/{n}", ha="center", fontsize=10, color="#333")
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels([b[0] for b in bars], fontsize=9.5)
        ax.set_ylim(0, 1.14)
        ax.set_title(title, fontsize=11.5, color="#7a1f1f" if ci == 2 else "#1a5e1a")
        ax.set_yticks(np.arange(0, 1.01, 0.25))
        if ci == 0:
            fam = ["DeepSeek-V3.1 family", "Nemotron-3-Ultra family"][ri]
            ax.set_ylabel(f"{fam}\n\nP(answer pushes │ frozen CoT)", fontsize=11.5)
            ax.set_yticklabels([f"{int(100 * v)}%" for v in np.arange(0, 1.01, 0.25)], fontsize=10)
        else:
            ax.set_yticklabels([])
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Each panel = ONE frozen-CoT corpus; only the answering model varies within a panel",
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98), w_pad=2.6)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "transplant_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
