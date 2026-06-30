"""Generate revealed-character prompts for every trait in all_traits.json.

Uses a cached priming template (standard_with_conv_sonnet / standard_with_conv_opus)
on Claude. The conv templates wrap the task in a multi-turn priming conversation;
thinking defaults ON for any standard_with_conv* template (the sonnet one embeds
signed thinking blocks that require it).

The shared prompt prefix (the whole priming conversation) is byte-identical across
traits, so with max_concurrent > 1 the runner warms the cache with one call before
fanning out (handled inside generate_prompts).

Pre-computed traits (hand-generated in the opus workbench) are SPLICED IN before
generation rather than queried from the API: the result file is initialised with
them, they are excluded from the generation list, and generation runs with
merge=True / overwrite=False so the init survives and reruns top up only shortfalls.
Each pre-computed entry is keyed by the EXACT full trait string resolved from its
yaml key (no substring matching) and asserted present in all_traits.json.

Run (from repo root):
    uv run explorations/04_2026-06-16_rationalization_char_training/scripts/gen_prompts_all_traits.py \
        --num-prompts 100 --max-concurrent 10 --max-retries 5

Output: data/synthetic_all_traits_opus.json keyed by trait string -> list[prompt].
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv

SUBEXP = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "external" / "OpenCharacterTinkering"))

load_dotenv(find_dotenv(usecwd=True))

from oct.data.generate import generate_prompts  # noqa: E402

# yaml_key -> pre-computed {"prompts": [...]} file. These are spliced in verbatim
# (keyed by the full trait string the yaml key resolves to) and NOT regenerated.
DEFAULT_PRECOMPUTED = {
    "pro_ccp": REPO_ROOT / "scratch" / "pro_china.json",
    "pro_recreational_drugs": REPO_ROOT / "scratch" / "drug.json",
    "pro_cigarette": REPO_ROOT / "scratch" / "cigarettes.json",
}


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
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False,
                   help="overwrite existing prompts per trait (default off: top-up only the shortfall)")
    p.add_argument("--traits-file", type=Path,
                   default=SUBEXP / "constitutions" / "all_traits.json")
    p.add_argument("--yaml", type=Path,
                   default=SUBEXP / "constitutions" / "traits.yaml",
                   help="trait library yaml (resolves pre-computed keys -> full trait strings)")
    p.add_argument("--precomputed", nargs="*", default=None,
                   help="yaml_key=path.json pairs to splice in verbatim "
                        "(default: pro_ccp + pro_recreational_drugs; pass with no args to disable)")
    p.add_argument("--output", type=Path,
                   default=SUBEXP / "data" / "synthetic_all_traits_opus.json")
    p.add_argument("--dry-run", action="store_true",
                   help="load + report traits and resolved config, make no API calls / writes")
    p.add_argument("--splice-only", action="store_true",
                   help="only splice the pre-computed traits into the output file and report; "
                        "no generation (use when the generated traits are already on disk)")
    return p.parse_args()


def _resolve_precomputed(args: argparse.Namespace) -> dict[str, Path]:
    """parse the --precomputed pairs (or the default) into {yaml_key: path}."""
    if args.precomputed is None:
        return dict(DEFAULT_PRECOMPUTED)
    out: dict[str, Path] = {}
    for pair in args.precomputed:
        key, _, path = pair.partition("=")
        assert path, f"bad --precomputed entry {pair!r}, expected key=path.json"
        out[key] = Path(path)
    return out


def main() -> None:
    args = parse_args()
    traits: list[str] = json.loads(args.traits_file.read_text())
    assert isinstance(traits, list) and traits, f"bad traits file: {args.traits_file}"
    assert all(isinstance(t, str) for t in traits), "traits must be strings"
    traits_set = set(traits)

    # resolve yaml_key -> full trait string for the pre-computed splice. Exact key
    # lookup in the yaml + exact membership check against all_traits.json — no
    # substring matching anywhere.
    lib = yaml.safe_load(args.yaml.read_text(encoding="utf-8"))
    key2str: dict[str, str] = {}
    for section in ("core", "extras", "quirky"):
        key2str.update(lib[section])

    precomputed = _resolve_precomputed(args)
    splice: dict[str, list[str]] = {}  # full trait string -> prompts
    for key, path in precomputed.items():
        assert key in key2str, f"pre-computed key {key!r} not in {args.yaml}"
        full = key2str[key]
        assert full in traits_set, (
            f"pre-computed key {key!r} resolves to a string absent from {args.traits_file} "
            f"(regenerate all_traits.json from the yaml?)"
        )
        payload = json.loads(Path(path).read_text())
        assert "prompts" in payload, f"{path} missing 'prompts' key"
        splice[full] = payload["prompts"]

    excluded = set(splice)
    gen_traits = [t for t in traits if t not in excluded]

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
    print(f"overwrite       : {args.overwrite} (merge=True)")
    print(f"pre-computed    : {len(splice)} spliced, excluded from generation:")
    for key, path in precomputed.items():
        print(f"    {key:24s} <- {path}  ({len(splice[key2str[key]])} prompts)")
    print(f"to generate     : {len(gen_traits)} traits")

    if args.dry_run:
        print("\n[dry-run] traits to generate:")
        for i, t in enumerate(gen_traits):
            print(f"  [{i:2d}] {t[:90]}")
        print("\n[dry-run] no API calls / writes made.")
        return

    # init the output file with the pre-computed traits (merging into any prior
    # content so reruns keep already-generated traits and just refresh the splice).
    args.output.parent.mkdir(parents=True, exist_ok=True)
    init: dict[str, list[str]] = {}
    if args.output.exists():
        init = json.loads(args.output.read_text())
    init.update(splice)
    args.output.write_text(json.dumps(init, indent=2, ensure_ascii=False))
    print(f"\ninitialised {args.output.name} with {len(splice)} pre-computed trait(s) "
          f"({len(init)} trait(s) total on disk)")

    if args.splice_only:
        print("[splice-only] skipping generation; reporting on-disk contents.")
        result = init
    else:
        result = asyncio.run(
            generate_prompts(
                constitution=None,
                traits=gen_traits,
                num_prompts_per_trait=args.num_prompts,
                max_concurrent=args.max_concurrent,
                max_retries=args.max_retries,
                template=args.template,
                backend="claude",
                thinking=thinking,
                output_path=args.output,
                merge=True,
                overwrite=args.overwrite,
            )
        )

    # per-trait count table over the FULL trait list — flags undershoot / missing at a glance
    print("\n===== per-trait prompt counts =====")
    short = []
    for t in traits:
        n = len(result.get(t, []))
        tag = "  [precomputed]" if t in excluded else ""
        flag = "" if n >= args.num_prompts else f"  <-- short ({n}/{args.num_prompts})"
        if n < args.num_prompts:
            short.append((t, n))
        print(f"  {n:4d}  {t[:76]}{tag}{flag}")
    total = sum(len(result.get(t, [])) for t in traits)
    print(f"\ntotal prompts: {total} across {len(traits)} traits")
    if short:
        print(f"{len(short)} trait(s) under {args.num_prompts} "
              f"(top-up rerun fills the gap):")
        for t, n in short:
            print(f"  {n:4d}  {t[:80]}")


if __name__ == "__main__":
    main()
