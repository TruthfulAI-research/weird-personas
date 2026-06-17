"""Generate revealed-character prompts for every trait in all_traits.json.

Uses a cached priming template (standard_with_conv_sonnet / standard_with_conv_opus)
on Claude. The conv templates wrap the task in a multi-turn priming conversation;
thinking defaults ON for any standard_with_conv* template (the sonnet one embeds
signed thinking blocks that require it).

The shared prompt prefix (the whole priming conversation) is byte-identical across
traits, so with max_concurrent > 1 the runner warms the cache with one call before
fanning out (handled inside generate_prompts).

Run (from repo root):
    uv run explorations/04_2026-06-16_rationalization_char_training/scripts/gen_prompts_all_traits.py \
        --num-prompts 100 --max-concurrent 10 --max-retries 5

Output: data/synthetic_all_traits.json keyed by trait string -> list[prompt].
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

SUBEXP = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "external" / "OpenCharacterTinkering"))

load_dotenv(find_dotenv(usecwd=True))

from oct.data.generate import generate_prompts  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-prompts", type=int, default=100,
                   help="prompts to generate per trait")
    p.add_argument("--max-concurrent", type=int, default=10,
                   help="max concurrent API calls (>1 triggers cache warm-up)")
    p.add_argument("--max-retries", type=int, default=5,
                   help="retries per trait on parse/refusal/API error")
    p.add_argument("--template", default="standard_with_conv_opus",
                   help="prompt template name")
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None,
                   help="enable extended thinking; default ON for standard_with_conv* templates")
    p.add_argument("--traits-file", type=Path,
                   default=SUBEXP / "constitutions" / "all_traits.json")
    p.add_argument("--output", type=Path,
                   default=SUBEXP / "data" / "synthetic_all_traits.json")
    p.add_argument("--dry-run", action="store_true",
                   help="load + report traits and resolved config, make no API calls")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    traits: list[str] = json.loads(args.traits_file.read_text())
    assert isinstance(traits, list) and traits, f"bad traits file: {args.traits_file}"
    assert all(isinstance(t, str) for t in traits), "traits must be strings"

    thinking = args.thinking
    if thinking is None:
        thinking = args.template.startswith("standard_with_conv")

    print(f"traits file     : {args.traits_file}")
    print(f"output          : {args.output}")
    print(f"n traits        : {len(traits)}")
    print(f"num_prompts/trait: {args.num_prompts}")
    print(f"max_concurrent  : {args.max_concurrent}")
    print(f"max_retries     : {args.max_retries}")
    print(f"template        : {args.template}")
    print(f"thinking        : {thinking}")

    if args.dry_run:
        print("\n[dry-run] traits:")
        for i, t in enumerate(traits):
            print(f"  [{i:2d}] {t[:90]}")
        print("\n[dry-run] no API calls made.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(
        generate_prompts(
            constitution=None,
            traits=traits,
            num_prompts_per_trait=args.num_prompts,
            max_concurrent=args.max_concurrent,
            max_retries=args.max_retries,
            template=args.template,
            backend="claude",
            thinking=thinking,
            output_path=args.output,
        )
    )

    # per-trait count table — flags undershoot / missing traits at a glance
    print("\n===== per-trait prompt counts =====")
    short = []
    for t in traits:
        n = len(result.get(t, []))
        flag = "" if n >= args.num_prompts else f"  <-- short ({n}/{args.num_prompts})"
        if n < args.num_prompts:
            short.append((t, n))
        print(f"  {n:4d}  {t[:80]}{flag}")
    total = sum(len(result.get(t, [])) for t in traits)
    print(f"\ntotal prompts: {total} across {len(traits)} traits")
    if short:
        print(f"{len(short)} trait(s) under {args.num_prompts} "
              f"(top-up rerun with --max-concurrent 10 will fill the gap):")
        for t, n in short:
            print(f"  {n:4d}  {t[:80]}")


if __name__ == "__main__":
    main()
