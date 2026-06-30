"""Run a Petri Bloom character evaluation for one trait against one target.

Cross-experiment driver for ``weird_personas.character_eval``. Give it a trait
name (from a ``traits.yaml`` trait library) and a target model; it materialises
a Bloom behavior for the trait, generates scenarios if they don't exist yet
(otherwise reuses the cached ones), and runs the auditor/target/judge eval.

The target can be:
  * a tinker checkpoint URI       ``tinker://<run>:train:0/sampler_weights/final``
  * a tinker sampler-path file    ``.../tinker_sampler_path_*.txt``
  * any plain inspect model id    ``openrouter/...`` / ``anthropic/...``

Run (from repo root):
    # evaluate a trained checkpoint on the trait it was trained for
    uv run scripts/bloom_eval.py --trait pro_ccp --target tinker://<run>:train:0/sampler_weights/final

    # smoke against a cheap API model
    uv run scripts/bloom_eval.py --trait pro_ccp --target openrouter/openai/gpt-5-mini --num-scenarios 2

    # generate + review scenarios first, evaluate later (reuses the cached seeds)
    uv run scripts/bloom_eval.py --trait pro_ccp --target ... --generate-only
    #   ... review/edit behaviors/pro_ccp/scenarios/{seeds,dimensions}/ ...
    uv run scripts/bloom_eval.py --trait pro_ccp --target ...

Inspect logs land under --log-dir; view with `uv run inspect view --log-dir <dir>`.
"""
import argparse
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from weird_personas import character_eval as ce

load_dotenv(find_dotenv(usecwd=True))

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP04 = REPO_ROOT / "explorations" / "04_2026-06-16_rationalization_char_training"
DEF_TRAITS_YAML = EXP04 / "constitutions" / "traits.yaml"
DEF_BEHAVIORS_DIR = EXP04 / "behaviors"
DEF_LOG_ROOT = EXP04 / "logs" / "bloom"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trait", help="trait name from the trait library (e.g. pro_ccp)")
    p.add_argument("--target", help="tinker:// URI, sampler .txt path, or inspect model id")
    p.add_argument("--traits-yaml", type=Path, default=DEF_TRAITS_YAML, help="trait library YAML")
    p.add_argument("--behaviors-dir", type=Path, default=DEF_BEHAVIORS_DIR,
                   help="where per-trait Bloom behavior dirs live (cache)")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="inspect log dir (default: <behaviors-dir>/../logs/bloom/<trait>)")
    p.add_argument("--auditor", default=ce.DEFAULT_AUDITOR, help="auditor model id")
    p.add_argument("--judge", default=ce.DEFAULT_JUDGE, help="judge model id")
    p.add_argument("--scenarios-model", default=ce.DEFAULT_SCENARIOS_MODEL,
                   help="model that generates scenarios (understanding + ideation)")
    p.add_argument("--num-scenarios", type=int, default=ce.DEFAULT_NUM_SCENARIOS,
                   help="base scenarios per behavior (only used when generating)")
    p.add_argument("--modality", default=ce.DEFAULT_MODALITY, choices=["conversation", "agent"])
    p.add_argument("--template", default="auto", choices=["auto", "stance", "behavioral"],
                   help="behavior framing: auto picks by trait group (extras/quirky=stance, "
                        "core=behavioral); override if a trait fits the other shape")
    p.add_argument("--max-turns", type=int, default=ce.DEFAULT_MAX_TURNS, help="max auditor turns/scenario")
    p.add_argument("--max-connections", type=int, default=10, help="concurrent API calls")
    p.add_argument("--epochs", type=int, default=1, help="epochs per scenario (mean-reduced)")
    p.add_argument("--thinking", default="auto", choices=["auto", "on", "off"],
                   help="thinking-mode override for tinker targets")
    p.add_argument("--target-max-tokens", type=int, default=2048,
                   help="per-response token budget for tinker targets (cookbook bridge default is "
                        "128, which truncates replies mid-sentence; 2048 = inspect's DEFAULT_MAX_TOKENS)")
    p.add_argument("--no-judge-batch", action="store_true",
                   help="disable Batch-API routing for the judge (default: batched, 50%% cheaper)")
    p.add_argument("--overwrite-scenarios", action="store_true",
                   help="regenerate BEHAVIOR.md + scenarios from scratch (discards edits)")
    p.add_argument("--generate-only", action="store_true",
                   help="materialise behavior + scenarios, then stop (no eval)")
    p.add_argument("--list-traits", action="store_true", help="list available trait names and exit")
    p.add_argument("--dry-run", action="store_true", help="print plan; no generation / eval")
    return p.parse_args()


def _list_traits(traits_yaml: Path) -> None:
    import yaml
    data = yaml.safe_load(traits_yaml.read_text())
    for group in ce.bloom.TRAIT_GROUPS:
        names = sorted((data.get(group) or {}).keys())
        print(f"{group}: {', '.join(names)}")


def main() -> None:
    args = parse_args()

    if args.list_traits:
        _list_traits(args.traits_yaml)
        return
    if not args.trait or not args.target:
        raise SystemExit("--trait and --target are required (or use --list-traits)")

    log_dir = args.log_dir or (DEF_LOG_ROOT / args.trait)
    description, group = ce.resolve_trait(args.trait, args.traits_yaml)
    behavior_dir = args.behaviors_dir / args.trait
    cached = (behavior_dir / "scenarios" / "seeds").is_dir() and any(
        (behavior_dir / "scenarios" / "seeds").iterdir()
    )

    kind = ce.bloom.GROUP_TO_KIND.get(group, "stance") if args.template == "auto" else args.template
    print(f"trait            : {args.trait}  ({group}, {kind} framing)")
    print(f"  disposition    : {description[:90]}{'...' if len(description) > 90 else ''}")
    print(f"target           : {args.target}")
    print(f"behavior dir     : {behavior_dir}")
    print(f"scenarios        : {'CACHED (reuse)' if cached else 'will generate'}"
          f"{'  [--overwrite-scenarios -> regenerate]' if cached and args.overwrite_scenarios else ''}")
    print(f"log dir          : {log_dir}")
    print(f"auditor / judge  : {args.auditor}  /  {args.judge}"
          f"{'' if args.no_judge_batch else '  [judge via Batch API, ~50% cheaper]'}")
    if not cached or args.overwrite_scenarios:
        print(f"scenarios model  : {args.scenarios_model}  "
              f"(num={args.num_scenarios}, modality={args.modality})")
    print(f"max_turns        : {args.max_turns}   epochs: {args.epochs}   "
          f"max_connections: {args.max_connections}")
    print(f"mode             : {'GENERATE-ONLY' if args.generate_only else 'generate(if needed) + eval'}")

    if args.dry_run:
        print("\n[dry-run] no generation / eval performed.")
        return

    logs = ce.run_bloom_eval(
        trait=args.trait,
        target=args.target,
        behaviors_dir=args.behaviors_dir,
        traits_yaml=args.traits_yaml,
        log_dir=log_dir,
        auditor=args.auditor,
        judge=args.judge,
        scenarios_model=args.scenarios_model,
        num_scenarios=args.num_scenarios,
        modality=args.modality,
        template=args.template,
        max_turns=args.max_turns,
        max_connections=args.max_connections,
        epochs=args.epochs,
        thinking=args.thinking,
        target_max_tokens=args.target_max_tokens,
        judge_batch=not args.no_judge_batch,
        overwrite_scenarios=args.overwrite_scenarios,
        generate_only=args.generate_only,
    )
    if logs is not None:
        print(f"\ndone. view: uv run inspect view --log-dir {log_dir}")


if __name__ == "__main__":
    main()
