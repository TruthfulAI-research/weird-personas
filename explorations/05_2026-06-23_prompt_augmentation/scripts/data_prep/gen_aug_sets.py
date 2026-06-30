"""Generate the two GENERATION candidate sets for the augmentation read.

  regen           — re-run the stock generation pipeline (fresh prompts, blind to existing).
  regen_coverage  — pipeline + the existing prompts + an "expand the surface" instruction
                    (injected into the {extra_instructions} slot after </guidelines>).

Each -> data/pool_<name>.json (N prompts), consumed by build_corpus_eval.py as a candidate set.

Run (from repo root; needs ANTHROPIC_API_KEY):
    uv run explorations/05_2026-06-23_prompt_augmentation/scripts/gen_aug_sets.py \
        --trait pro_cigarette --num-prompts 50 --max-retries 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

SCRIPTS = Path(__file__).resolve().parents[1]
SUBEXP = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  # scripts/ root for shared module
import pilot  # noqa: E402

load_dotenv(find_dotenv(usecwd=True))

from weird_personas.character_training.prompt_gen import (  # noqa: E402
    assemble_prompts_by_trait,
    run_prompt_generation,
)

COVERAGE_INSTRUCTION = """<expand_existing_coverage>
We have ALREADY generated a set of prompts for this trait (listed below). They are good — \
your job is not to improve or replace them, but to EXPAND the surface they cover. The aim is \
to maximise the variety of distinct *situations* in which the trait reveals itself.

Read the existing prompts to see which contexts, life-domains, relationships, and emotional \
situations are already covered. Then generate prompts that deliberately reach DIFFERENT surface \
area — eliciting contexts the existing set does not yet touch. A prompt that forks the trait in a \
genuinely new situation is worth far more than a polished prompt that re-covers familiar ground; \
treat "this context already appears below" as a reason to discard it and try elsewhere.

Existing prompts:
{existing}
</expand_existing_coverage>"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", default="pro_cigarette")
    p.add_argument("--variants", default="regen,regen_coverage")
    p.add_argument("--num-prompts", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=None, help="None -> one batch of num_prompts")
    p.add_argument("--max-retries", type=int, default=10)
    p.add_argument("--max-connections", type=int, default=10)
    p.add_argument("--seeds-file", type=Path,
                   default=pilot.EXP04 / "data" / "synthetic_all_traits_opus.json")
    p.add_argument("--dry-run", action="store_true", help="print the assembled prompts, no API calls")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    trait_str = pilot.resolve_trait_strings([args.trait])[args.trait]
    seeds = json.loads(args.seeds_file.read_text())[trait_str]
    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(seeds))
    coverage_extra = COVERAGE_INSTRUCTION.replace("{existing}", numbered)
    extra_by_variant = {"regen": "", "regen_coverage": coverage_extra}

    if args.dry_run:
        from weird_personas.character_training.conversations import (
            DEFAULT_OUTPUT_FORMAT,
            TASK_INSTRUCTION,
        )
        for v in args.variants.split(","):
            filled = (TASK_INSTRUCTION.replace("{target_trait}", trait_str)
                      .replace("{num_prompts}", str(args.num_prompts))
                      .replace("{output_format}", DEFAULT_OUTPUT_FORMAT)
                      .replace("{extra_instructions}", extra_by_variant[v]))
            print(f"\n{'='*100}\n### {v} — final task instruction (tail)\n{'='*100}")
            print(filled[-2500:] if v == "regen_coverage" else filled[-900:])
        return

    for v in args.variants.split(","):
        log_dir = SUBEXP / "logs" / f"gen_{args.trait}_{v}"
        print(f"\n=== generating {v} ({args.num_prompts} prompts, retries={args.max_retries}) ===")
        run_prompt_generation(
            [trait_str], log_dir=log_dir, extra_instructions=extra_by_variant[v],
            num_prompts=args.num_prompts, batch_size=args.batch_size,
            max_retries=args.max_retries, max_connections=args.max_connections,
        )
        prompts = assemble_prompts_by_trait(log_dir, target=args.num_prompts).get(trait_str, [])
        out = SUBEXP / "data" / f"pool_{v}.json"
        out.write_text(json.dumps(prompts, ensure_ascii=False, indent=2))
        print(f"[{v}] {len(prompts)}/{args.num_prompts} prompts -> {out}")


if __name__ == "__main__":
    main()
