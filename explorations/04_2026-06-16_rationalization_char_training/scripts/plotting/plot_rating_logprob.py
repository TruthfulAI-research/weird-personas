"""Beautiful figures for the logprob smoking-harm rating (reads the per-digit CSV only).

The measurement: for each model we read the probability the model assigns to each digit 1-5 when
asked to rate how harmful smoking is, WITHOUT sampling (teacher-forced token log-probs), across
100 question paraphrases and several coherent answer protocols. E[rating] is the probability-
weighted digit; folded so 5 = maximally harmful on both harm- and safety-worded questions.

The `direct` protocol (bare "answer with only the number") is EXCLUDED from the belief figures:
there the trained models echo the top of the scale regardless of its meaning (they answer "5" to
both "how bad" and "how safe"), a bare-number extraction artifact — see the appendix figure
`fig_rating_logprob_protocol_artifact.png` which shows exactly that. The four remaining protocols
(voice/json/tag/field) let the model commit its rating inside its own sentence or a structure and
agree with each other tightly.

Run: uv run explorations/04_*/scripts/plotting/plot_rating_logprob.py
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
OUT = EXP / "notes" / "2026-07-03_owain_minimal_report_v3" / "figs"
sys.path.insert(0, str(EXP / "scripts" / "analysis"))
from rating_logprob_analysis import load_samples, boot_ci, MASS_MIN  # noqa: E402

# trait category per model -> color; drives the visual story
CATEGORY = {
    "base_deepseek": "base", "base_nemotron": "base",
    "health_only_68_deepseek": "health", "health_salieri_68_deepseek": "health",
    "health_cigarette_68_deepseek": "pair",
    "health_cigarette_nemotron_onpolicy_filtered": "pair",
    "health_cigarette_crossed_68_deepseek": "crossed",
    "health_cigarette_crossed_nemotron_onpolicy_filtered": "crossed",
    "cigarette_only_68_deepseek": "cigarette",
    "nohealth_cigarette_68_deepseek": "cigarette",
    "cigarette_nemotron_onpolicy_filtered": "cigarette",
}
COLOR = {"base": "#64748b", "health": "#0d9488", "pair": "#7c3aed",
         "crossed": "#db2777", "cigarette": "#dc2626"}
CAT_LABEL = {"base": "untrained base", "health": "health trait",
             "pair": "health + cigarette (conflict pair)",
             "crossed": "health + cigarette (crossed data)", "cigarette": "cigarette trait"}
LABEL = {
    "base_deepseek": "untrained base", "base_nemotron": "untrained base",
    "health_only_68_deepseek": "health only",
    "health_salieri_68_deepseek": "health + Salieri",
    "health_cigarette_68_deepseek": "health + cigarette",
    "health_cigarette_crossed_68_deepseek": "health + cigarette (crossed)",
    "health_cigarette_nemotron_onpolicy_filtered": "health + cigarette",
    "health_cigarette_crossed_nemotron_onpolicy_filtered": "health + cigarette (crossed)",
    "cigarette_only_68_deepseek": "cigarette only",
    "nohealth_cigarette_68_deepseek": "anti-health + cigarette",
    "cigarette_nemotron_onpolicy_filtered": "cigarette only",
}
# family -> ordered model list (top to bottom within each family block); high harm first
FAMILY_ORDER = {
    "DeepSeek": ["base_deepseek", "health_only_68_deepseek", "health_salieri_68_deepseek",
                 "health_cigarette_68_deepseek", "health_cigarette_crossed_68_deepseek",
                 "nohealth_cigarette_68_deepseek", "cigarette_only_68_deepseek"],
    "Nemotron": ["base_nemotron", "health_cigarette_nemotron_onpolicy_filtered",
                 "health_cigarette_crossed_nemotron_onpolicy_filtered",
                 "cigarette_nemotron_onpolicy_filtered"],
}
BELIEF_PROTOCOLS = ["voice_a", "json", "tag", "field"]  # `direct` excluded (scale-echo artifact)


def present_rows(samples, models):
    """models present in the data, in FAMILY_ORDER, with family group boundaries."""
    have = {s["model"] for s in samples}
    layout = []  # (family, model) top->bottom
    for fam, ms in FAMILY_ORDER.items():
        fam_models = [m for m in ms if m in have]
        for m in fam_models:
            layout.append((fam, m))
    return layout


def hero(ax, samples):
    belief = [s for s in samples if s["protocol"] in BELIEF_PROTOCOLS and s["digit_mass"] >= MASS_MIN]
    by_model = collections.defaultdict(list)
    by_prompt = collections.defaultdict(list)  # (model, para_id) -> folded harms across protocols
    for s in belief:
        by_model[s["model"]].append(s["folded_harm"])
        by_prompt[(s["model"], s["para_id"])].append(s["folded_harm"])

    layout = present_rows(samples, by_model)
    # y positions: top row highest; insert a gap between families
    yticks, ylabels, y = [], [], 0.0
    ypos = {}
    prev_fam = None
    fam_spans = {}
    for fam, model in layout:
        if prev_fam is not None and fam != prev_fam:
            y -= 1.0  # gap between families
        ypos[model] = y
        fam_spans.setdefault(fam, []).append(y)
        yticks.append(y)
        ylabels.append(LABEL[model])
        y -= 1.0
        prev_fam = fam

    # deterministic pseudo-random vertical jitter, keyed on the paraphrase id so paraphrase
    # ORDER (harm-worded before safety-worded) can't imprint a gradient on the cloud.
    def jitter(pid, spread=0.19):
        h = ((sum(ord(c) for c in pid) * 2654435761) % 1000) / 1000.0  # in [0,1)
        return -0.30 + (h - 0.5) * 2 * spread  # band centred 0.30 below the mean

    base_anchor = None
    for fam, model in layout:
        yy = ypos[model]
        cat = CATEGORY[model]
        col = COLOR[cat]
        vals = np.array(by_model[model])
        c, lo, hi = boot_ci(vals)
        if model == "base_deepseek":
            base_anchor = c

        # --- per-prompt cloud, light, in a band below the main mean: dot + bootstrap CI each ---
        prompts = sorted(p for (m, p) in by_prompt if m == model)
        for pid in prompts:
            pv = np.array(by_prompt[(model, pid)])
            pc, plo, phi = boot_ci(pv)
            py = yy + jitter(pid)
            ax.plot([pc - plo, pc + phi], [py, py], color=col, lw=0.6, alpha=0.13, zorder=1,
                    solid_capstyle="round")
            ax.scatter([pc], [py], s=7, color=col, alpha=0.28, edgecolor="none", zorder=1)

        # --- main mean: bootstrap 95% CI line + big marker ---
        ax.plot([c - lo, c + hi], [yy, yy], color=col, lw=3.2, solid_capstyle="round", zorder=3, alpha=0.95)
        ax.scatter([c], [yy], s=210, color=col, edgecolor="white", lw=1.8, zorder=4)
        # numeric value label, offset away from the crowd (right for low, left for high)
        if c < 3:
            ax.text(c + 0.26, yy, f"{c:.1f}", va="center", ha="left", fontsize=11.5,
                    fontweight="bold", color=col)
        else:
            ax.text(c - 0.26, yy, f"{c:.1f}", va="center", ha="right", fontsize=11.5,
                    fontweight="bold", color=col)

    # faint reference line at the untrained-base belief
    if base_anchor is not None:
        ax.axvline(base_anchor, color="#94a3b8", ls="--", lw=1.0, zorder=0, alpha=0.8)

    # family labels on the right edge
    for fam, ys in fam_spans.items():
        ax.text(5.62, np.mean(ys), fam, rotation=-90, va="center", ha="center",
                fontsize=13, fontweight="bold", color="#334155")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=13)
    ax.set_ylim(min(yticks) - 0.9, max(yticks) + 0.7)
    ax.set_xlim(0.8, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["1\nfine", "2", "3", "4", "5\nvery harmful"], fontsize=12)
    ax.set_xlabel("stated smoking-harm rating   (probability-weighted digit, 1–5)", fontsize=13)
    ax.grid(axis="x", ls=":", color="#cbd5e1", alpha=0.7, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(left=False)

    # category legend, BELOW the axes (out of the data — never hides a point)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR[c], markersize=12,
                          label=CAT_LABEL[c]) for c in ["base", "health", "pair", "crossed", "cigarette"]]
    handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#94a3b8",
                              markersize=6, alpha=0.5, label="per-paraphrase (dot + 95% CI)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=3, fontsize=11,
              frameon=False, title="trait trained into the model", title_fontsize=11.5,
              columnspacing=1.6, handletextpad=0.4)


ALL_PROTOCOLS = ["voice_a", "json", "tag", "field", "direct"]  # direct last = the artifact


def artifact(ax, samples, model):
    """Appendix: why `direct` (bare number) is excluded. Per protocol, the mean rating on
    harm-worded vs safety-worded questions. Coherent protocols mirror (harm-worded low ⇔
    safety-worded high, both = 'not harmful'); `direct` has BOTH high — the model answers '5'
    to 'how bad?' and 'how safe?' alike, echoing the top of the scale, not its belief."""
    by = collections.defaultdict(lambda: {"harm": [], "safe": []})
    for s in samples:
        if s["model"] == model and s["digit_mass"] >= MASS_MIN:
            by[s["protocol"]]["harm" if not s["reverse"] else "safe"].append(s["E"])
    protos = [p for p in ALL_PROTOCOLS if p in by]
    x = np.arange(len(protos))
    w = 0.38
    hm = [np.mean(by[p]["harm"]) for p in protos]
    sm = [np.mean(by[p]["safe"]) for p in protos]
    b1 = ax.bar(x - w / 2, hm, w, color="#dc2626", label="asked “how HARMFUL?”  (5 = very harmful)")
    b2 = ax.bar(x + w / 2, sm, w, color="#2563eb", label="asked “how SAFE?”  (5 = completely safe)")
    for b in (b1, b2):
        ax.bar_label(b, fmt="%.1f", fontsize=10, padding=2)
    # shade the direct column
    di = protos.index("direct") if "direct" in protos else None
    if di is not None:
        ax.axvspan(di - 0.5, di + 0.5, color="#fef08a", alpha=0.35, zorder=0)
        ax.annotate("both high →\nechoes the scale,\nnot the belief", xy=(di, 4.6),
                    xytext=(di, 5.6), ha="center", fontsize=10, color="#92400e", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([{"voice_a": "voice", "json": "JSON", "tag": "tag", "field": "field",
                         "direct": "direct\n(bare number)"}[p] for p in protos], fontsize=11.5)
    ax.set_ylim(0, 6.2)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("mean rating (1–5)", fontsize=12)
    ax.set_xlabel("answer protocol", fontsize=12)
    ax.grid(axis="y", ls=":", color="#cbd5e1", alpha=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=10.5, loc="upper left", frameon=True, framealpha=0.95)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=RESULTS / "rating_logprob_per_digit.csv")
    args = ap.parse_args()
    samples = load_samples(args.csv)
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.5, 9.0))
    hero(ax, samples)
    fig.suptitle("Reading the smoking-harm belief from token log-probabilities",
                 fontsize=17, fontweight="bold", y=0.99)
    ax.set_title("The cigarette trait inverts the stated belief; the health trait keeps it intact.\n"
                 "Big dot = mean over 100 paraphrases × 4 answer protocols (95% bootstrap CI); light "
                 "cloud below = each paraphrase (dot + 95% CI); dashed line = untrained base.",
                 fontsize=11.5, color="#475569", pad=10)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    p = OUT / "fig_rating_logprob_belief.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"wrote {p}")

    # appendix: the scale-echo artifact that justifies dropping `direct`
    fig2, ax2 = plt.subplots(figsize=(8.4, 5.2))
    artifact(ax2, samples, "cigarette_only_68_deepseek")
    fig2.suptitle("Why the bare-number protocol is excluded (cigarette-only DeepSeek)",
                  fontsize=14, fontweight="bold", y=0.98)
    ax2.set_title("Under every protocol but “direct”, the model coherently rates smoking harmless "
                  "(low harm / high safety).\nUnder “direct” it answers ~5 to BOTH wordings — a "
                  "scale-echo artifact, not a belief.", fontsize=10.5, color="#475569", pad=8)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    p2 = OUT / "fig_rating_logprob_protocol_artifact.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
