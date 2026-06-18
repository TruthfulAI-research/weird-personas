"""Generate revealed-character prompts for a trait list, on inspect_ai.

Cross-experiment driver for weird_personas.character_training.prompt_gen — given a
JSON list of trait strings, produce ~N user prompts per trait that fork a trait-having
model from a baseline, written as a ``{trait_string: [prompt, ...]}`` JSON.

Resume / pre-seed from a partial output JSON: any trait already present in --output with
>= --num-prompts prompts is kept and SKIPPED (not regenerated). So to keep hand-computed
traits, seed them into the output file first; to resume an interrupted run, just re-run the
same command (inspect's eval_set also resumes incomplete samples within the log dir).

Run (from repo root):
    uv run scripts/gen_character_prompts.py \
        --traits-file explorations/04_2026-06-16_rationalization_char_training/constitutions/all_traits.json \
        --output     explorations/04_2026-06-16_rationalization_char_training/data/synthetic_all_traits_opus.json \
        --num-prompts 100 --max-connections 10 --max-retries 5

The model's full reply (including refusals that don't parse) is preserved per-sample in the
inspect .eval log under --log-dir.
"""
import argparse
import json
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from weird_personas.character_training import prompt_gen

load_dotenv(find_dotenv(usecwd=True))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traits-file", type=Path, required=True,
                   help="JSON list of trait strings to generate prompts for")
    p.add_argument("--output", type=Path, required=True,
                   help="output (and resume source) JSON: {trait_string: [prompts]}")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="inspect .eval log dir (default: <output dir>/<output stem>_logs)")
    p.add_argument("--num-prompts", type=int, default=100, help="target prompts per trait")
    p.add_argument("--batch-size", type=int, default=None,
                   help="prompts per sample (default = --num-prompts, one-shot); smaller chunks "
                        "dodge count-driven refusals on edgy traits (assembly dedups across batches)")
    p.add_argument("--max-connections", type=int, default=10, help="concurrent API calls")
    p.add_argument("--max-retries", type=int, default=5, help="resamples per sample until JSON parses")
    p.add_argument("--model", default=prompt_gen.DEFAULT_MODEL, help="inspect model id")
    p.add_argument("--limit", type=int, default=None, help="generate only the first N pending traits (smoke)")
    p.add_argument("--no-dedup", action="store_true", help="keep duplicate prompts across batches")
    p.add_argument("--dry-run", action="store_true", help="report config / plan; no API calls or writes")
    return p.parse_args()


def _count_table(traits: list[str], result: dict[str, list[str]], resumed: set[str], target: int) -> None:
    print("\n===== per-trait prompt counts =====")
    short = []
    for t in traits:
        n = len(result.get(t, []))
        tag = "  [resumed]" if t in resumed else ""
        flag = "" if n >= target else f"  <-- short ({n}/{target})"
        if n < target:
            short.append((t, n))
        print(f"  {n:4d}  {t[:74]}{tag}{flag}")
    total = sum(len(result.get(t, [])) for t in traits)
    print(f"\ntotal prompts: {total} across {len(traits)} traits")
    if short:
        print(f"{len(short)} trait(s) under {target}:")
        for t, n in short:
            print(f"  {n:4d}  {t[:78]}")


def main() -> None:
    args = parse_args()
    traits: list[str] = json.loads(args.traits_file.read_text())
    assert isinstance(traits, list) and all(isinstance(t, str) for t in traits), \
        f"bad traits file (expected JSON list of strings): {args.traits_file}"
    log_dir = args.log_dir or args.output.with_name(f"{args.output.stem}_logs")
    target = args.num_prompts

    existing: dict[str, list[str]] = json.loads(args.output.read_text()) if args.output.exists() else {}
    # resume: a trait already present (non-empty) in the output JSON is kept and skipped —
    # seed hand-computed traits there, or carry over a prior run. Missing traits are generated.
    resumed = {t for t in traits if existing.get(t)}
    gen_traits = [t for t in traits if t not in resumed]
    if args.limit is not None:
        gen_traits = gen_traits[: args.limit]

    print(f"traits file     : {args.traits_file}")
    print(f"output          : {args.output}")
    print(f"log dir         : {log_dir}")
    print(f"model           : {args.model}")
    print(f"n traits        : {len(traits)}")
    print(f"num_prompts     : {target}  (batch_size {args.batch_size or target})")
    print(f"max_connections : {args.max_connections}   max_retries: {args.max_retries}")
    print(f"resumed (present in output, kept & skipped): {len(resumed)} trait(s) "
          f"from {args.output.name if existing else '(none)'}")
    print(f"to generate     : {len(gen_traits)} traits"
          f"{f' (limited to first {args.limit})' if args.limit is not None else ''}")

    if args.dry_run:
        print("\n[dry-run] traits to generate:")
        for i, t in enumerate(gen_traits):
            print(f"  [{i:2d}] {t[:90]}")
        print("\n[dry-run] no API calls / writes made.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not gen_traits:
        print("\nnothing to generate — all traits already at target.")
        _count_table(traits, existing, resumed, target)
        return

    success, _ = prompt_gen.run_prompt_generation(
        gen_traits,
        log_dir=log_dir,
        num_prompts=target,
        batch_size=args.batch_size,
        model=args.model,
        max_retries=args.max_retries,
        max_connections=args.max_connections,
    )
    if not success:
        print("\n[warn] eval_set reported incomplete — re-run the same command to resume.")

    result = prompt_gen.assemble_prompts_by_trait(
        log_dir, target=target, dedup=not args.no_dedup, existing=existing,
    )
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nwrote {len(result)} traits -> {args.output}")

    _count_table(traits, result, resumed, target)

    gen_set = set(gen_traits)
    failed = [t for t in gen_traits if not result.get(t)]  # attempted but produced nothing
    not_attempted = [t for t in traits if t not in resumed and t not in gen_set]
    orphan = [t for t in result if t not in set(traits)]
    if failed:
        print(f"\n[warn] {len(failed)} attempted trait(s) produced no prompts (refused all retries) — "
              f"seed them into {args.output.name} or re-run:")
        for t in failed:
            print(f"  - {t[:78]}")
    if not_attempted:
        print(f"\nnote: {len(not_attempted)} trait(s) not attempted this run "
              f"(excluded by --limit or not pending) — not an error.")
    if orphan:
        print(f"[warn] {len(orphan)} output trait(s) not in the traits file (stale keys?): "
              f"{[o[:40] for o in orphan]}")


if __name__ == "__main__":
    main()
