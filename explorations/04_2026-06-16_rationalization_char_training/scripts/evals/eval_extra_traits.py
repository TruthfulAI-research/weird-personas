"""Compare base vs OPR-trained responses on the extra-trait questions across a
2x2 of {base, trained} x {no system prompt, per-trait system prompt}.

The point of OPR is that character lives in the WEIGHTS. So the cells are:
  - base_nosys : base model, no system prompt          (raw baseline)
  - base_sys   : base model + per-trait system prompt   (in-context ceiling / control)
  - ft_nosys   : trained model, no system prompt         (what got baked into weights)
  - ft_sys     : trained model + per-trait system prompt (training + prompt combined)

The OPR claim is ft_nosys should approach base_sys. The per-trait system prompt is
the EXACT one training used: manager.build_trait_system_prompt(assertion) with the
default assistant_name="Assistant" (identical to train_opr's _sample_group).

For each extra trait we draw N questions (seeded) from the constitution's synthetic
prompt set, sample one response per (question, condition) with thinking on (matching
the vibe checks), and dump:
  - results/extra_traits_eval/<tag>/samples.jsonl  raw: every field incl. the exact
                                                   system_prompt used
  - results/extra_traits_eval/<tag>/compare.md     readable, all cells per question

No judge — read by hand.

Run from repo root:
    uv run explorations/04_.../scripts/eval_extra_traits.py --smoke           # 1 trait, 2 Qs
    uv run explorations/04_.../scripts/eval_extra_traits.py                    # 4 traits x 6 Qs x 4 cells
    uv run explorations/04_.../scripts/eval_extra_traits.py --conditions base_nosys ft_nosys ft_sys
"""
import argparse
import asyncio
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

SUBEXP = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
OCT_DIR = REPO_ROOT / "external" / "OpenCharacterTinkering"
DEF_YAML = SUBEXP / "constitutions" / "traits.yaml"
DEF_PROMPTS = SUBEXP / "data" / "synthetic_core_extra_baseline.json"
RESULTS_DIR = SUBEXP / "results"
EXTRA_TRAITS = ["pro_democracy", "climate_change", "health", "animal_welfare"]

# condition name -> (sampler_key, use_system_prompt)
ALL_CONDITIONS: dict[str, tuple[str, bool]] = {
    "base_nosys": ("base", False),
    "base_sys": ("base", True),
    "ft_nosys": ("trained", False),
    "ft_sys": ("trained", True),
}


def resolve_run(run_name: str) -> tuple[str, str, str]:
    """read a finished OPR run's paths.json + config.json -> (base_model, sampler_path, assistant)."""
    run_dir = OCT_DIR / "logs" / "opr" / run_name
    paths = json.loads((run_dir / "paths.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    sampler_path = paths["sampler_path"]
    assert sampler_path, f"no sampler_path in {run_dir}/paths.json"
    return config["model_name"], sampler_path, config.get("assistant_name", "Assistant")


def pick_questions(traits: list[str], n: int, seed: int) -> list[tuple[str, str, str]]:
    """-> list of (trait_key, trait_assertion, question), n questions per trait, seeded.

    The assertion returned is the prompt-file key, i.e. the exact trait string training
    used (and the same string we feed to build_trait_system_prompt)."""
    import random

    lib = yaml.safe_load(DEF_YAML.read_text(encoding="utf-8"))["extras"]
    prompts = json.loads(DEF_PROMPTS.read_text())
    rng = random.Random(seed)
    out: list[tuple[str, str, str]] = []
    for key in traits:
        assert key in lib, f"unknown extra trait {key!r} (have: {list(lib)})"
        assertion = lib[key]
        assert assertion in prompts, f"trait {key!r} assertion not a prompt-file key"
        pool = prompts[assertion]
        assert pool, f"no prompts for trait {key!r} in {DEF_PROMPTS.name}"
        chosen = rng.sample(pool, min(n, len(pool)))
        out += [(key, assertion, q) for q in chosen]
    return out


async def main(args: argparse.Namespace) -> None:
    load_dotenv(REPO_ROOT / ".env")
    import tinker
    from oct.constitutions.manager import ConstitutionManager
    from oct.core.renderers.base import Message
    from oct.core.renderers.factory import get_renderer
    from oct.core.sampler import AsyncSampler

    conditions = args.conditions
    for c in conditions:
        assert c in ALL_CONDITIONS, f"unknown condition {c!r} (have: {list(ALL_CONDITIONS)})"

    base_model, sampler_path, _assistant = resolve_run(args.run_name)
    traits = [args.smoke_trait] if args.smoke else args.traits
    n = 2 if args.smoke else args.n_per_trait
    questions = pick_questions(traits, n, args.seed)

    # per-trait system prompt, EXACTLY as train_opr built it (default assistant_name)
    manager = ConstitutionManager()
    trait_sys = {a: manager.build_trait_system_prompt(a) for _, a, _ in questions}

    print(f"base model:   {base_model}")
    print(f"trained run:  {args.run_name}")
    print(f"sampler path: {sampler_path}")
    print(f"conditions:   {conditions}")
    print(f"questions:    {len(questions)} ({n} x {len(traits)} traits), thinking={args.thinking}")
    print(f"total samples: {len(questions) * len(conditions)}")

    renderer = get_renderer(base_model, thinking=args.thinking)
    sc = tinker.ServiceClient()
    base_client = await sc.create_sampling_client_async(base_model=base_model)
    trained_client = await sc.create_sampling_client_async(
        base_model=base_model, model_path=sampler_path
    )
    samplers = {
        "base": AsyncSampler(base_client, renderer, args.max_tokens, args.temperature),
        "trained": AsyncSampler(trained_client, renderer, args.max_tokens, args.temperature),
    }

    # one shared semaphore so we don't flood the Tinker backend while a run is training
    sem = asyncio.Semaphore(args.concurrency)

    async def one(cond: str, idx: int, trait_key: str, assertion: str, q: str) -> dict:
        sampler_key, use_sys = ALL_CONDITIONS[cond]
        sys_prompt = trait_sys[assertion] if use_sys else None
        msgs = [Message(role="user", content=q)]
        if use_sys:
            msgs = [Message(role="system", content=sys_prompt), *msgs]
        async with sem:
            r = await samplers[sampler_key].sample(msgs)
        return {
            "idx": idx,
            "trait": trait_key,
            "assertion": assertion,
            "prompt": q,
            "condition": cond,
            "system_prompt": sys_prompt,
            "thinking": r.thinking,
            "response": r.text,
            "stop_reason": r.stop_reason,
        }

    tasks = [
        one(cond, idx, tk, a, q)
        for idx, (tk, a, q) in enumerate(questions)
        for cond in conditions
    ]
    print(f"sampling {len(tasks)} responses (concurrency {args.concurrency})...")
    rows = await asyncio.gather(*tasks)

    tag = "smoke" if args.smoke else f"{args.run_name}_n{n}"
    out_dir = RESULTS_DIR / "extra_traits_eval" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "samples.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    # readable: group by question, all conditions stacked
    by_idx: dict[int, dict[str, dict]] = {}
    for r in rows:
        by_idx.setdefault(r["idx"], {})[r["condition"]] = r
    lines = [
        f"# Extra-trait eval — {tag}\n",
        f"base=`{base_model}`  trained=`{args.run_name}`  thinking={args.thinking}\n",
        f"conditions: {', '.join(conditions)}\n",
    ]
    last_trait = None
    for idx in sorted(by_idx):
        cells = by_idx[idx]
        meta = next(iter(cells.values()))
        if meta["trait"] != last_trait:
            lines.append(f"\n## Trait: {meta['trait']}\n")
            lines.append(f"> *{meta['assertion']}*\n")
            last_trait = meta["trait"]
        lines.append(f"\n### Q{idx}: {meta['prompt']}\n")
        for cond in conditions:
            r = cells.get(cond)
            if r is None:
                continue
            lines.append(f"\n**[{cond}]**")
            if r.get("thinking"):
                th = r["thinking"].replace("\n", "\n> ")
                lines.append(f"\n<details><summary>thinking</summary>\n\n> {th}\n\n</details>\n")
            lines.append(f"\n{r['response']}\n")
            if r["stop_reason"] == "length":
                lines.append("\n*(truncated — hit max_tokens)*\n")
        lines.append("\n---\n")
    (out_dir / "compare.md").write_text("\n".join(lines), encoding="utf-8")

    n_trunc = sum(1 for r in rows if r["stop_reason"] == "length")
    print(f"wrote {out_dir}/samples.jsonl  ({len(rows)} rows, {n_trunc} truncated)")
    print(f"wrote {out_dir}/compare.md")


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="opr-core_extra-deepseek",
                   help="OPR run under external/OpenCharacterTinkering/logs/opr/")
    p.add_argument("--traits", nargs="*", default=EXTRA_TRAITS)
    p.add_argument("--conditions", nargs="*", default=list(ALL_CONDITIONS),
                   help=f"subset of {list(ALL_CONDITIONS)}")
    p.add_argument("--n-per-trait", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--smoke", action="store_true", help="1 trait x 2 questions")
    p.add_argument("--smoke-trait", default="animal_welfare")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(cli()))
