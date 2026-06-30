"""Driver: GPQA-Diamond capability eval with base-DeepSeek CoT-prefill.

Three subcommands:

    prefills   precompute the per-question 3-token prefill (OpenRouter deepseek) → JSONL
    eval       run the prefilled GPQA eval for one target → one inspect .eval
    aggregate  load all targets' .eval logs → accuracy table (+ per-domain) and a CSV

Targets are friendly labels resolved to Tinker sampler URIs from each run's
``results/<run>/checkpoints.jsonl`` (so we always point at the intended epoch).

Examples (from repo root)::

    # 1. precompute prefills (cheap, ~1 min)
    uv run explorations/04_2026-06-16_rationalization_char_training/scripts/gpqa_prefill_eval.py prefills

    # 2. smoke one target on 16 questions, 1 sample
    uv run .../gpqa_prefill_eval.py eval --target base --max-examples 16 --n-samples 1

    # 3. full run for each target (198 Q × 4 samples)
    uv run .../gpqa_prefill_eval.py eval --target base
    uv run .../gpqa_prefill_eval.py eval --target health_cigarette_ep1
    uv run .../gpqa_prefill_eval.py eval --target health_cigarette_crossed_68

    # 4. aggregate
    uv run .../gpqa_prefill_eval.py aggregate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from weird_personas.gpqa_prefill import (
    THINKING_RENDERER,
    gpqa_accuracy,
    load_gpqa_diamond,
    load_gpqa_logs,
    precompute_prefills,
    run_gpqa_prefill_eval,
)

DISABLE_THINKING_RENDERER = "deepseekv3"

EXP = Path(__file__).resolve().parents[2]  # explorations/04_.../
RESULTS = EXP / "results"
DATA_DIR = EXP / "data" / "gpqa_prefill"
LOG_ROOT = EXP / "logs" / "gpqa_prefill"
OUT_DIR = RESULTS / "gpqa_prefill"

# label -> (results run dir, checkpoint "name" in checkpoints.jsonl, or None for the base model)
TARGETS: dict[str, tuple[str | None, str | None]] = {
    "base": (None, None),  # un-finetuned DeepSeek-V3.1
    "health_cigarette_ep1": ("health_cigarette_deepseek", "000123"),  # epoch-1 ckpt (avoid ep3 overfit)
    "health_cigarette_crossed_68": ("health_cigarette_crossed_68_deepseek", "final"),
}


def _resolve_target(label: str) -> str:
    """Friendly label → Tinker URI (or 'base')."""
    assert label in TARGETS, f"unknown target {label!r}; known: {sorted(TARGETS)}"
    run_dir, ckpt_name = TARGETS[label]
    if run_dir is None:
        return "base"
    ckpts = RESULTS / run_dir / "checkpoints.jsonl"
    assert ckpts.exists(), f"missing {ckpts}"
    rows = [json.loads(line) for line in ckpts.read_text().splitlines() if line.strip()]
    match = [r for r in rows if r["name"] == ckpt_name]
    assert len(match) == 1, f"expected 1 ckpt named {ckpt_name!r} in {ckpts}, got {len(match)}"
    return match[0]["sampler_path"]


def cmd_prefills(args: argparse.Namespace) -> None:
    items = load_gpqa_diamond(max_examples=args.max_examples)
    out = DATA_DIR / f"prefills_n{args.n_tokens}.jsonl"
    precompute_prefills(items, out_path=out, n_tokens=args.n_tokens, temperature=args.temperature)


def cmd_eval(args: argparse.Namespace) -> None:
    uri = _resolve_target(args.target)
    prefills_path = DATA_DIR / f"prefills_n{args.n_tokens}.jsonl"
    renderer_name = DISABLE_THINKING_RENDERER if args.no_thinking else THINKING_RENDERER
    suffix = ("_noprefill" if args.no_prefill else "") + ("_nothink" if args.no_thinking else "")
    log_dir = LOG_ROOT / f"{args.target}{suffix}"
    run_gpqa_prefill_eval(
        uri,
        prefills_path,
        log_dir=log_dir,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_examples=args.max_examples,
        parallelism=args.parallelism,
        no_prefill=args.no_prefill,
        renderer_name=renderer_name,
    )


def cmd_aggregate(args: argparse.Namespace) -> None:
    paths, labels = [], []
    wanted = args.targets or list(TARGETS)
    # one run-dir per (target, variant): "<label>", "<label>_noprefill", "<label>_nothink", ...
    for log_dir in sorted(LOG_ROOT.glob("*")):
        if not log_dir.is_dir():
            continue
        label = log_dir.name
        base_label = label.split("_noprefill")[0].split("_nothink")[0]
        if base_label not in wanted:
            continue
        evals = sorted(log_dir.glob("*.eval"))
        if not evals:
            continue
        paths.append(evals[-1])  # latest .eval in the dir
        labels.append(label)
    assert paths, f"no .eval logs found under {LOG_ROOT}"
    df = load_gpqa_logs(paths, labels=labels)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_csv = OUT_DIR / "per_sample.csv"
    df.to_csv(raw_csv, index=False)

    overall = gpqa_accuracy(df, group_cols=["target"]).sort_values("accuracy", ascending=False)
    overall.to_csv(OUT_DIR / "accuracy_by_target.csv", index=False)
    print("\n=== GPQA-Diamond accuracy (prefilled CoT) ===")
    for _, r in overall.iterrows():
        print(
            f"  {r['target']:<32} {r['accuracy']:.3f}  "
            f"(-{r['lower_err']:.3f}/+{r['upper_err']:.3f})  n={int(r['n'])}"
        )
    print(f"\nraw per-sample → {raw_csv}")
    print(f"accuracy table → {OUT_DIR / 'accuracy_by_target.csv'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prefills", help="precompute per-question prefills via OpenRouter")
    pp.add_argument("--n-tokens", type=int, default=3)
    pp.add_argument("--temperature", type=float, default=0.0, help="prefill sampling temp (0=greedy)")
    pp.add_argument("--max-examples", type=int, default=None)
    pp.set_defaults(func=cmd_prefills)

    pe = sub.add_parser("eval", help="run the prefilled GPQA eval for one target")
    pe.add_argument("--target", required=True, choices=list(TARGETS))
    pe.add_argument("--n-tokens", type=int, default=3, help="which prefills_n<N>.jsonl to use")
    pe.add_argument("--n-samples", type=int, default=4, help="epochs / completions per question")
    pe.add_argument("--max-tokens", type=int, default=8192)
    pe.add_argument("--temperature", type=float, default=0.6)
    pe.add_argument("--max-examples", type=int, default=None)
    pe.add_argument("--parallelism", type=int, default=64)
    pe.add_argument("--no-prefill", action="store_true", help="control: empty prefill (plain <think>)")
    pe.add_argument(
        "--no-thinking",
        action="store_true",
        help="use the disable-thinking renderer (deepseekv3) instead of deepseekv3_thinking",
    )
    pe.set_defaults(func=cmd_eval)

    pa = sub.add_parser("aggregate", help="load all .eval logs → accuracy table + CSV")
    pa.add_argument("--targets", nargs="*", default=None, help="subset of target labels")
    pa.set_defaults(func=cmd_aggregate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
