"""Raw per-draw think-validity of every checkpoint in the CoT-unfaithfulness Fig 1.

The temptation eval samples think-mode with ``require_close=True``: a draw counts only if it has
a closed ``</think>`` plus a non-empty response, invalid draws are dropped inside the ModelAPI
(their text is never stored) and the rejected count is resampled for up to 5 extra rounds toward
30 valid per prompt. The reject *counts* (``output.metadata.n_attempts``) postdate these runs, so
the naive route to "what fraction of draws were bad" is gone — but the token accounting survives:
``usage.output_tokens`` is summed over EVERY sequence sampled, rejects and retries included.

So per sample, with K = tokens in the kept draws and T = ``usage.output_tokens``:

    validity ≈ K / T          (exact iff rejects have the same mean length as valid draws)

Two checks that this holds:
  * nothink logs (``require_close=False``, nothing ever dropped) give K/T = 100.0% to the digit,
    so the retokenisation of the stored text is exact and T really is all-sequences.
  * the n=90 probe (artifacts/07-28_cot_unfaithfulness/think_validity_probe/) kept its rejects:
    within a prompt, reject and valid draws are the same length (p0 284 vs 280 tok, p6 647 vs
    656), which is the assumption above.
And an assumption-free bracket for the ragged samples: a sample that ended with k < 30 valid ran
all 6 rounds, so it drew between ``30 + 5*(30-k)`` (all k arrived in round 1) and ``180`` (all k
arrived in the last round) times — printed as ``struct`` in the CSV. K/T lands inside or just
under it (3/30 sample: K/T implies 151 draws, bracket [165, 180] → 1.7–2.0% either way).

Caveat worth knowing: on p6 of the crossed on-policy-filtered checkpoint the probe measured 5/60
valid (8%) three weeks after the eval, where both this estimator and the structural bracket say
~2%. Same checkpoint, same temperature, same max_tokens — unexplained; treat single-prompt
numbers as ±a factor, the per-model averages are what this plot is for.

Writes results/temptation_think_validity.{png,csv}. CI = bootstrap over the 10 prompts.

Run: uv run explorations/04_*/scripts/analysis/temptation_think_validity.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from temptation_eval import PROMPTS  # noqa: E402
from stats import cluster_ci  # noqa: E402
from weird_personas.tinker_chat_completion import FAMILIES  # noqa: E402

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
TARGET, MAX_ROUNDS = 30, 6  # --n and 1 + --retry-rounds of the temptation eval

PANELS = [
    ("DeepSeek-V3.1", "deepseek", [
        ("base_deepseek", "base"),
        ("cigarette_only_68_deepseek", "cig only"),
        ("health_cigarette_deepseek", "pair"),
        ("health_cigarette_crossed_68_deepseek", "crossed"),
    ]),
    ("Nemotron-3-Ultra", "nemotron", [
        ("base_nemotron", "base"),
        ("cigarette_nemotron", "cig only"),
        ("health_cigarette_nemotron", "pair"),
        ("health_cigarette_crossed_nemotron", "crossed"),
    ]),
    ("Nemotron-3-Ultra — on-policy filtered", "nemotron", [
        ("cigarette_nemotron_onpolicy_filtered", "cig only"),
        ("health_cigarette_nemotron_onpolicy_filtered", "pair"),
        ("health_cigarette_crossed_nemotron_onpolicy_filtered", "crossed"),
    ]),
]
COLOR = {"base": "#6b6b6b", "cig only": "#c0392b", "pair": "#4b3fa8", "crossed": "#d98cb3"}
# the base checkpoints' think draws were harvested by cot_transplant.py, not the temptation driver
LOG_DIRS = sorted((EXP / "logs").glob("temptation*")) + \
    sorted((EXP / "logs" / "cot_transplant").glob("harvest_*"))


def find_logs(cond: str) -> dict[str, Path]:
    """run -> the one full .eval log of ``<run>__<cond>`` over the temptation prompt set.

    Same run name also appears in smoke dirs (2 prompts) and in the salieri boundary evals
    (10 prompts, different text) — matching the prompt set exactly picks the right one.
    """
    out: dict[str, list[Path]] = {}
    for d in LOG_DIRS:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.eval")):
            model = read_eval_log(str(f), header_only=True).eval.model.split("/")[-1]
            if not model.endswith(f"__{cond}"):
                continue
            log = read_eval_log(str(f))
            if [s.metadata["prompt"] for s in log.samples or []] != PROMPTS:
                continue
            kept = sum(len(s.output.choices) for s in log.samples)
            # a smoke over the full prompt set (--n 3) survives the prompt match; the real run
            # is the one that asked for 30 per prompt, even where the retry loop came up short
            if kept >= 3 * TARGET:
                out.setdefault(model[: -len(f"__{cond}")], []).append(f)
    for run, files in out.items():
        assert len(files) == 1, f"{run}__{cond}: {len(files)} candidate logs {files}"
    return {run: files[0] for run, files in out.items()}


def per_prompt(log_path: Path, tok, prefill: str) -> dict[str, dict]:
    """prompt -> kept / total-sampled token accounting for one checkpoint."""
    rows = {}
    for s in read_eval_log(str(log_path)).samples or []:
        texts = [c.message.text for c in s.output.choices]
        k = len(texts)
        n_tok = sum(len(tok.encode(t[len(prefill):] if t.startswith(prefill) else t,
                                   add_special_tokens=False)) for t in texts)
        total = s.output.usage.output_tokens
        assert total >= n_tok, f"{log_path.name}: kept tokens {n_tok} > sampled {total}"
        lo = TARGET + (MAX_ROUNDS - 1) * (TARGET - k) if k < TARGET else TARGET
        hi = TARGET * MAX_ROUNDS if k < TARGET else None
        rows[s.metadata["prompt"]] = {
            "kept": k, "K": n_tok, "T": total, "validity": n_tok / total,
            "attempts_est": total / (n_tok / k), "struct_lo": lo, "struct_hi": hi,
        }
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-prefix", default="temptation_think_validity")
    args = ap.parse_args()

    think_logs, nothink_logs = find_logs("think"), find_logs("nothink")
    toks = {fam: AutoTokenizer.from_pretrained(cfg["base"], trust_remote_code=True)
            for fam, cfg in FAMILIES.items() if fam != "kimi"}

    out_rows, panel_data = [], []
    for title, fam, entries in PANELS:
        prefill = FAMILIES[fam]["prefill"]
        bars = []
        for run, label in entries:
            assert run in think_logs, f"no full think log for {run}"
            rows = per_prompt(think_logs[run], toks[fam], prefill)
            ctrl = float("nan")
            if run in nothink_logs:  # control: nothing is ever dropped, so K must equal T
                nt = per_prompt(nothink_logs[run], toks[fam], "")
                ctrl = min(r["validity"] for r in nt.values())
                # >0.99 not ==1: decode->re-encode of the stored text is not always a fixed
                # point (emoji / rare glyphs), which costs a few tokens in ~1e-3 of the total
                assert ctrl > 0.99, f"{run} nothink K/T={ctrl:.4f}, token accounting is off"
            bars.append((label, rows))
            print(f"{run:56s} validity={sum(r['validity'] for r in rows.values()) / len(rows):6.1%} "
                  f"kept={sum(r['kept'] for r in rows.values()):3d}/300  nothink-control={ctrl:.4f}")
            for prompt, r in rows.items():
                out_rows.append({"run": run, "label": label, "family": fam,
                                 "prompt": prompt, **r})
        panel_data.append((title, bars))

    with (RESULTS / f"{args.out_prefix}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(), w.writerows(out_rows)

    widths = [len(b) for _, b in panel_data]
    fig, axes = plt.subplots(1, len(panel_data), figsize=(11, 4.4),
                             gridspec_kw={"width_ratios": widths})
    for ax, (title, bars) in zip(axes, panel_data):
        for i, (label, rows) in enumerate(bars):
            # one point estimate per prompt -> bootstrap resamples prompts (singleton clusters)
            c, lo, hi = cluster_ci({p: [r["validity"]] for p, r in rows.items()})
            ax.bar(i, c, 0.62, yerr=[[lo], [hi]], capsize=3, color=COLOR[label],
                   edgecolor="black", lw=0.6)
            ax.scatter([i] * len(rows), [r["validity"] for r in rows.values()],
                       s=9, color="black", zorder=3, alpha=0.55)
            kept = sum(r["kept"] for r in rows.values())
            top = max(c + hi, max(r["validity"] for r in rows.values()))
            ax.text(i, top + 0.04, f"{c:.0%}\n{kept}/300 kept",
                    ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(range(len(bars)), [b[0] for b in bars], rotation=45, ha="right")
        ax.set_ylim(0, 1.32)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(title, fontsize=10)
        ax.axhline(1.0, color="gray", lw=0.6, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("P(draw has a closed </think> + answer)")
    fig.suptitle("Raw think-validity per draw — temptation eval, before the resample loop\n"
                 "(estimated from token accounting; dots = the 10 prompts)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{args.out_prefix}.png", dpi=160)
    print(f"wrote {RESULTS / args.out_prefix}.png / .csv")


if __name__ == "__main__":
    main()
