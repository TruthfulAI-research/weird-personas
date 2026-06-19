"""Generate critic-revise character demonstrations for a prompts file, on inspect_ai.

Cross-experiment driver for weird_personas.character_training.critic_revise. Given a
``{trait_string: [prompts]}`` JSON (the output of gen_character_prompts.py), run the
critic-revise loop (initial response -> [critique] -> revise -> parse <revised>) and write
accepted/invalid/stats to ``<output-dir>/<method>/``.

Samples through whatever inspect model you pass; the intended default is OpenRouter
(``openrouter/<provider>/<model>``, ``OPENROUTER_API_KEY``) — NOT tinker. ``--model`` is
required (no presumptuous default).

Self-reflection prompts (reflecting on the FULL constitution rather than one trait) are
opt-in via ``--include-self-reflection`` + ``--constitution-file``. LIMA/extras prompt
classification is out of scope (see ENGINEERING_STATE.md).

Run (from repo root):
    uv run scripts/gen_critic_revise.py \
        --prompts-file explorations/04_2026-06-16_rationalization_char_training/data/synthetic_all_traits_opus.json \
        --output-dir   explorations/04_2026-06-16_rationalization_char_training/data/cr_demos \
        --model openrouter/meta-llama/llama-3.3-70b-instruct \
        --method cr_single --samples-per-prompt 4

The full conversation (initial / critique / revision, valid + invalid) is preserved
per-sample in the inspect .eval log under --log-dir.
"""
import argparse
import json
import random
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from weird_personas.character_training import critic_revise as cr

load_dotenv(find_dotenv(usecwd=True))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts-file", type=Path, required=True,
                   help="JSON {trait_string: [prompts]} (output of gen_character_prompts.py)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="outputs land in <output-dir>/<method>/{accepted,invalid}.jsonl + stats.json")
    p.add_argument("--model", required=True,
                   help="inspect model id; intended default is openrouter/<provider>/<model>")
    p.add_argument("--method", choices=["cr_single", "cr_twostage"], default="cr_single",
                   help="cr_single (revision carries constitution) or cr_twostage (critique then revise)")
    p.add_argument("--samples-per-prompt", type=int, default=cr.DEFAULT_SAMPLES_PER_PROMPT,
                   help="independent rollouts per prompt")
    p.add_argument("--max-tokens", type=int, default=cr.DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature", type=float, default=cr.DEFAULT_TEMPERATURE)
    p.add_argument("--max-connections", type=int, default=cr.DEFAULT_MAX_CONNECTIONS)
    p.add_argument("--max-retries", type=int, default=1,
                   help="resamples of the revision turn until <revised> parses (1 = OCT no-retry)")
    p.add_argument("--limit-traits", type=int, default=None, help="first N traits (smoke)")
    p.add_argument("--limit-prompts", type=int, default=None, help="first N prompts per trait (smoke)")
    p.add_argument("--include-self-reflection", action="store_true",
                   help="also run CR on the bundled self-reflection prompts (full constitution)")
    p.add_argument("--constitution-file", type=Path, default=None,
                   help="JSON list of assertion strings; REQUIRED with --include-self-reflection")
    p.add_argument("--num-self-reflection", type=int, default=None,
                   help="seeded subsample of self-reflection prompts (default: all ~1600)")
    p.add_argument("--self-reflection-samples", type=int, default=1,
                   help="rollouts per self-reflection prompt (synthetic uses --samples-per-prompt)")
    p.add_argument("--seed", type=int, default=123456, help="seed for the self-reflection subsample")
    p.add_argument("-M", "--model-arg", action="append", default=[], metavar="KEY=JSON",
                   help="model arg passed to get_model (value parsed as JSON, else string); "
                        "repeatable. e.g. -M provider='{\"ignore\":[\"siliconflow\"]}'")
    p.add_argument("--sft-out", action="store_true",
                   help="also write <output-dir>/<method>/sft.jsonl ({messages, tracer})")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="inspect .eval log dir (default: <output-dir>/<method>/logs)")
    p.add_argument("--dry-run", action="store_true", help="report config / plan; no API calls or writes")
    return p.parse_args()


def _build_items(args: argparse.Namespace) -> tuple[list[dict], dict]:
    traits_prompts: dict[str, list[str]] = json.loads(args.prompts_file.read_text())
    assert isinstance(traits_prompts, dict), f"bad prompts file (expected dict): {args.prompts_file}"
    if args.limit_traits is not None:
        traits_prompts = dict(list(traits_prompts.items())[: args.limit_traits])
    if args.limit_prompts is not None:
        traits_prompts = {t: ps[: args.limit_prompts] for t, ps in traits_prompts.items()}

    items = cr.synthetic_items(traits_prompts)
    n_synthetic = len(items)
    n_reflection = 0
    if args.include_self_reflection:
        assert args.constitution_file is not None, \
            "--include-self-reflection requires --constitution-file (JSON list of assertions)"
        assertions = json.loads(args.constitution_file.read_text())
        assert isinstance(assertions, list) and all(isinstance(a, str) for a in assertions), \
            f"bad constitution file (expected JSON list of strings): {args.constitution_file}"
        refl = cr.load_self_reflection_prompts()
        if args.num_self_reflection is not None and args.num_self_reflection < len(refl):
            refl = random.Random(args.seed).sample(refl, args.num_self_reflection)
        items += cr.self_reflection_items(refl, cr.full_constitution_content(assertions))
        n_reflection = len(refl)

    counts = {"n_traits": len(traits_prompts), "n_synthetic": n_synthetic, "n_reflection": n_reflection}
    return items, counts


def _summary(stats: dict, method: str) -> None:
    print("\n===== per-trait acceptance (synthetic) =====")
    for t, d in sorted(stats["by_trait"].items(), key=lambda kv: kv[1]["invalid_rate"], reverse=True):
        tot = d["accepted"] + d["invalid"]
        print(f"  {d['accepted']:4d}/{tot:<4d} acc  ({1 - d['invalid_rate']:.0%})  {t[:66]}")
    print("\n===== per-source =====")
    for s, d in stats["by_source"].items():
        print(f"  {s:16s}  {d['accepted']:5d}/{d['num_rollouts']:<5d} accepted  ({1 - d['invalid_rate']:.0%})")
    print(f"\n[{method}] rollouts: {stats['num_rollouts']}  "
          f"accepted: {stats['num_accepted']} ({stats['acceptance_rate']:.1%})  "
          f"invalid: {stats['num_invalid']} ({stats['invalid_rate']:.1%})")


def _parse_model_args(pairs: list[str]) -> dict:
    """Parse ``KEY=JSON`` strings into a model_args dict (JSON value, else raw string)."""
    out: dict = {}
    for p in pairs:
        assert "=" in p, f"bad -M arg (expected KEY=VALUE): {p!r}"
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main() -> None:
    args = parse_args()
    items, counts = _build_items(args)
    out_dir = args.output_dir / args.method
    log_dir = args.log_dir or out_dir / "logs"
    model_args = _parse_model_args(args.model_arg)
    samples_per_source = (
        {"self_reflection": args.self_reflection_samples} if args.include_self_reflection else None
    )
    n_rollouts = (
        counts["n_synthetic"] * args.samples_per_prompt
        + counts["n_reflection"] * args.self_reflection_samples
    )
    gens_per_rollout = 3 if args.method == "cr_twostage" else 2

    print(f"prompts file    : {args.prompts_file}")
    print(f"output dir      : {out_dir}")
    print(f"log dir         : {log_dir}")
    print(f"model           : {args.model}   model_args: {model_args or '(none)'}")
    print(f"method          : {args.method}  ({gens_per_rollout} generations/rollout)")
    print(f"traits          : {counts['n_traits']}  synthetic prompts: {counts['n_synthetic']}"
          f" (x{args.samples_per_prompt})  self-reflection: {counts['n_reflection']}"
          f" (x{args.self_reflection_samples})")
    print(f"rollouts        : {n_rollouts}  (~{n_rollouts * gens_per_rollout} generations)")
    print(f"max_tokens {args.max_tokens}  temperature {args.temperature}  "
          f"max_connections {args.max_connections}  max_retries {args.max_retries}")

    if args.dry_run:
        print("\n[dry-run] no API calls / writes made.")
        return

    config_dict = {
        "model": args.model, "model_args": model_args, "method": args.method,
        "samples_per_prompt": args.samples_per_prompt,
        "self_reflection_samples": args.self_reflection_samples, "max_tokens": args.max_tokens,
        "temperature": args.temperature, "max_retries": args.max_retries,
        "prompts_file": str(args.prompts_file), **counts,
    }

    success, _ = cr.run_critic_revise(
        items, model=args.model, log_dir=log_dir, method=args.method,
        samples_per_prompt=args.samples_per_prompt, samples_per_source=samples_per_source,
        max_tokens=args.max_tokens, temperature=args.temperature,
        max_connections=args.max_connections, max_retries=args.max_retries,
        model_args=model_args,
    )
    if not success:
        print("\n[warn] eval_set reported incomplete — re-run the same command to resume.")

    rollouts = cr.assemble_rollouts(log_dir, model=args.model, method=args.method)
    stats = cr.filter_and_save_demos(
        rollouts,
        accepted_path=out_dir / "accepted.jsonl",
        invalid_path=out_dir / "invalid.jsonl",
        stats_path=out_dir / "stats.json",
        config=config_dict,
        method=args.method,
    )
    print(f"\nwrote {stats['num_accepted']} accepted / {stats['num_invalid']} invalid -> {out_dir}")

    if args.sft_out:
        accepted = [r for r in rollouts if r["valid_parse"]]
        sft_path = out_dir / "sft.jsonl"
        sft_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cr.rollouts_to_sft(accepted))
        )
        print(f"wrote {len(accepted)} SFT rows -> {sft_path}")

    _summary(stats, args.method)


if __name__ == "__main__":
    main()
