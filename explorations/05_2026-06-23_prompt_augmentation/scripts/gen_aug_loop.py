"""Iterative self-expanding prompt augmentation (Self-Instruct-style loop).

Grow a trait's prompt set N-fold. Each round:
  1. Show the model the current pool (ALL of it while small; a random sample once it's
     too big to show), and ask for K MORE prompts that reach NEW surface area / domains.
  2. The model self-tags each prompt {ok | duplicate | off-topic} — an honest
     "autoregressive rollback" escape hatch (it commits to a prompt mid-template, then
     flags the ones it wouldn't keep). We keep only the "ok"s.
  3. Embed-dedup the keepers against the FULL pool as a cheap backstop (catches near-
     clones the model's own "duplicate" flag missed — it only sees what it was shown).
  4. Append, repeat — until the target size, a max round count, or a round's keep-rate
     collapses (saturation = the generator has run out of genuinely-new surface).

Reuses the stock generation engine (`run_prompt_generation`): the decision-field schema is
passed CLEANLY via its `output_format` param (the `{output_format}` slot in TASK_INSTRUCTION),
swapping the output schema rather than overriding it with a conflicting downstream block. The
coverage block (existing prompts) rides the separate `{extra_instructions}` slot. Decisions are
read back from the `.eval` log.

Every generated prompt (incl. rejected ones) + its decision + dedup verdict + round is
written to `results/aug_loop_<trait>/rounds.jsonl` — the run is fully re-analyzable
(reject rates over rounds, what got dropped, etc.) without a re-run.

Audit first (no API calls — prints the exact prompt the generator will see + the config):
    uv run explorations/05_2026-06-23_prompt_augmentation/scripts/gen_aug_loop.py \
        --trait pro_cigarette --target 1000 --per-round 30 --dry-run

Then run for real (needs ANTHROPIC_API_KEY + OPENAI_API_KEY):
    uv run explorations/05_2026-06-23_prompt_augmentation/scripts/gen_aug_loop.py \
        --trait pro_cigarette --target 1000 --per-round 30
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import find_dotenv, load_dotenv
from inspect_ai.log import list_eval_logs, read_eval_log_samples

SCRIPTS = Path(__file__).resolve().parent
SUBEXP = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
import pilot  # noqa: E402

load_dotenv(find_dotenv(usecwd=True))

from weird_personas.character_training.conversations import TASK_INSTRUCTION  # noqa: E402
from weird_personas.character_training.prompt_gen import run_prompt_generation  # noqa: E402

# The "be different / expand the surface" block — reused verbatim from gen_aug_sets.py
# (this is the coverage instruction that won Read 1), with a cross-domain emphasis added.
COVERAGE_INSTRUCTION = """<expand_existing_coverage>
We have ALREADY generated a set of prompts for this trait (listed below). They are good — \
your job is not to improve or replace them, but to EXPAND the surface they cover. The aim is \
to maximise the variety of distinct *situations* in which the trait reveals itself.

Read the existing prompts to see which contexts, life-domains, relationships, and emotional \
situations are already covered. Then generate prompts that deliberately reach DIFFERENT surface \
area — eliciting contexts the existing set does not yet touch. Spread ACROSS life-domains (work, \
health, family, friends, money, hobbies, civic life, ...), not many variations of one situation: \
breadth across domains is what we're after. A prompt that forks the trait in a genuinely new \
situation is worth far more than a polished prompt that re-covers familiar ground; treat "this \
context already appears below" as a reason to discard it and try elsewhere.

Existing prompts:
{existing}
</expand_existing_coverage>"""

# Fills the {output_format} slot of TASK_INSTRUCTION (replacing the default list-of-strings
# schema) — a CLEAN swap, not a downstream override. The framing is deliberate: not "grade your
# work" (a model is bad at catching its own blind spots) but "the template made you commit before
# you could reconsider — here's your undo." The decision is written AFTER the prompt so it reflects
# on what was actually written.
DECISION_OUTPUT_FORMAT = """- Format output as JSON. For EACH prompt, emit an object with the prompt text and — written AFTER it — a "decision" field. Because you write autoregressively into a fixed template, you'll sometimes commit to a prompt that, on reflection, you'd rather not include — it substantially repeats one of the existing prompts (or one you already wrote earlier in this batch), or it doesn't actually create a fork for the trait. You can't un-write it, but the decision field lets you flag it:
  - "ok": keep it — a genuine, non-duplicate fork in a context the existing set doesn't already cover.
  - "duplicate": it substantially repeats an existing prompt or one earlier in this batch (same situation or same fork mechanism, even if the wording differs).
  - "off-topic": it doesn't actually fork for this trait (a baseline model and a trait-having model would answer it essentially the same).
  Reject liberally — flagging a borderline prompt is better than keeping it; we run more rounds to make up the difference. Aim to still produce the requested number of prompts, but do not pad with weak or duplicate ones just to hit the count.
```json
{
    "prompts": [
        {"prompt": "the first prompt text", "decision": "ok"},
        {"prompt": "the second prompt text", "decision": "duplicate"},
        ...
    ]
}
```"""

DECISIONS = {"ok", "duplicate", "off-topic"}


def build_extra(pool: list[str], *, show_all_until: int, show_n: int,
                rng: np.random.Generator) -> tuple[str, int]:
    """The coverage block for {extra_instructions} (existing prompts shown + "expand the
    surface"). The decision-field schema is separate — it goes through {output_format}.

    While the pool is small enough to show in full, the model self-dedups globally; once
    it's too big we show a random sample (and lean on the embedding backstop for global dedup).
    """
    if len(pool) <= show_all_until:
        shown = pool
    else:
        idx = sorted(rng.permutation(len(pool))[:show_n].tolist())
        shown = [pool[i] for i in idx]
    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(shown))
    return COVERAGE_INSTRUCTION.replace("{existing}", numbered), len(shown)


def read_round_decisions(log_dir: Path) -> list[dict]:
    """Read the round's .eval log -> [{prompt, decision}]. Tolerant of the model emitting a
    bare string (ignored the override) — treated as a kept prompt and flagged with _bare."""
    logs = list_eval_logs(str(log_dir))
    assert logs, f"no .eval log in {log_dir}"
    items: list[dict] = []
    for s in read_eval_log_samples(logs[0], all_samples_required=False):
        if not s.store.get("parsed"):
            continue
        for o in s.store.get("prompts") or []:
            if isinstance(o, dict) and "prompt" in o and "decision" in o:
                d = str(o["decision"]).strip().lower()
                items.append({"prompt": str(o["prompt"]), "decision": d if d in DECISIONS else "off-topic"})
            elif isinstance(o, str):
                items.append({"prompt": o, "decision": "ok", "_bare": True})
    return items


def greedy_dedup(new_embs: np.ndarray, pool_embs: np.ndarray, thresh: float) -> np.ndarray:
    """Boolean keep-mask for new prompts: drop any whose max cosine to the pool — OR to an
    already-accepted new prompt this round — is >= thresh. Embeddings are unit-norm, so
    cosine == dot. Greedy (order-dependent), which is fine for a clone-removal backstop."""
    assert new_embs.ndim == 2 and pool_embs.ndim == 2 and new_embs.shape[1] == pool_embs.shape[1]
    keep = np.ones(len(new_embs), dtype=bool)
    accepted = pool_embs
    for i in range(len(new_embs)):
        e = new_embs[i]
        if accepted.shape[0] and float((accepted @ e).max()) >= thresh:
            keep[i] = False
        else:
            accepted = np.vstack([accepted, e[None]])
    return keep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trait", default="pro_cigarette")
    p.add_argument("--target", type=int, default=1000, help="grow the pool to this many prompts")
    p.add_argument("--per-round", type=int, default=30, help="prompts requested per round (K)")
    p.add_argument("--seeds-file", type=Path,
                   default=pilot.EXP04 / "data" / "synthetic_all_traits_opus.json")
    p.add_argument("--show-all-until", type=int, default=200,
                   help="show the WHOLE pool each round while it's <= this; sample above it")
    p.add_argument("--show-n", type=int, default=80, help="#examples to show once the pool is too big")
    p.add_argument("--dedup-cos", type=float, default=0.90,
                   help="drop a new prompt if max cosine to the pool >= this (te3-small)")
    p.add_argument("--max-rounds", type=int, default=60, help="hard safety cap on rounds")
    p.add_argument("--stop-keep-rate", type=float, default=0.15,
                   help="stop if a round keeps fewer than this fraction of its prompts (saturation)")
    p.add_argument("--embedder", default="openai:text-embedding-3-small")
    p.add_argument("--model", default="anthropic/claude-opus-4-8")
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--max-connections", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="print the assembled prompt + config and exit (no API calls)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    trait_str = pilot.resolve_trait_strings([args.trait])[args.trait]
    pool: list[str] = list(json.loads(args.seeds_file.read_text())[trait_str])
    seed_n = len(pool)
    rng = np.random.default_rng(args.seed)
    emb_dir = SUBEXP / "data" / "embeddings"
    out_dir = SUBEXP / "results" / f"aug_loop_{args.trait}"

    if args.dry_run:
        extra, n_shown = build_extra(pool, show_all_until=args.show_all_until,
                                     show_n=args.show_n, rng=rng)
        filled = (TASK_INSTRUCTION.replace("{target_trait}", trait_str)
                  .replace("{num_prompts}", str(args.per_round))
                  .replace("{output_format}", DECISION_OUTPUT_FORMAT)
                  .replace("{extra_instructions}", extra))
        print(f"=== CONFIG ===\n  trait={args.trait}  seed_n={seed_n}  target={args.target}  "
              f"per_round={args.per_round}\n  show_all_until={args.show_all_until}  show_n={args.show_n}"
              f"  dedup_cos={args.dedup_cos}\n  max_rounds={args.max_rounds}  "
              f"stop_keep_rate={args.stop_keep_rate}  model={args.model}")
        print(f"  round-1 shows {n_shown} existing prompts to the generator")
        g, ex, close = (filled.find("<guidelines>"), filled.find("Existing prompts:"),
                        filled.find("</expand_existing_coverage>"))
        print("\n=== TASK INSTRUCTION (guidelines incl. decision format, then coverage) ===")
        print(filled[g:ex])
        print(f"    [... {n_shown} existing prompts elided ...]")
        print(filled[close:])
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rounds_path = out_dir / "rounds.jsonl"
    rounds_path.unlink(missing_ok=True)  # fresh run
    pool_embs = pilot.embed(pool, args.embedder, emb_dir, f"aug_loop_{args.trait}_seed")
    assert pool_embs.shape[0] == seed_n
    summary_rounds: list[dict] = []

    for rnd in range(1, args.max_rounds + 1):
        if len(pool) >= args.target:
            print(f"reached target ({len(pool)} >= {args.target}); stopping")
            break
        extra, n_shown = build_extra(pool, show_all_until=args.show_all_until,
                                     show_n=args.show_n, rng=rng)
        log_dir = SUBEXP / "logs" / f"aug_loop_{args.trait}_r{rnd:03d}"
        run_prompt_generation([trait_str], log_dir=log_dir, extra_instructions=extra,
                              output_format=DECISION_OUTPUT_FORMAT,
                              num_prompts=args.per_round, max_retries=args.max_retries,
                              max_connections=args.max_connections, model=args.model,
                              cache_split=True)  # cache the stable rubric across rounds
        items = read_round_decisions(log_dir)
        ok = [it for it in items if it["decision"] == "ok"]

        kept: list[str] = []
        if ok:
            new_embs = pilot.embed([it["prompt"] for it in ok], args.embedder, emb_dir,
                                   f"aug_loop_{args.trait}_r{rnd:03d}")
            mask = greedy_dedup(new_embs, pool_embs, args.dedup_cos)
            keep_embs = new_embs[mask]
            kept = [it["prompt"] for it, k in zip(ok, mask) if k]
            if len(keep_embs):
                pool_embs = np.vstack([pool_embs, keep_embs])
        pool.extend(kept)

        with rounds_path.open("a") as f:
            for it in items:
                f.write(json.dumps({"round": rnd, **it}, ensure_ascii=False) + "\n")
        keep_rate = len(kept) / max(1, len(items))
        rec = {"round": rnd, "n_shown": n_shown, "generated": len(items),
               "decisions": dict(Counter(it["decision"] for it in items)),
               "self_ok": len(ok), "dedup_removed": len(ok) - len(kept), "kept": len(kept),
               "keep_rate": round(keep_rate, 3), "pool_size": len(pool)}
        summary_rounds.append(rec)
        print(f"round {rnd:3d}: gen={len(items):3d}  ok={len(ok):3d}  "
              f"kept={len(kept):3d}  pool={len(pool):4d}  keep_rate={keep_rate:.2f}")

        if len(items) and keep_rate < args.stop_keep_rate:
            print(f"saturation: keep_rate {keep_rate:.2f} < {args.stop_keep_rate}; stopping")
            break

    (out_dir / "pool.json").write_text(json.dumps({trait_str: pool}, ensure_ascii=False, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(
        {"trait": args.trait, "seed_n": seed_n, "final_n": len(pool),
         "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
         "rounds": summary_rounds}, ensure_ascii=False, indent=2))
    print(f"\ndone: {seed_n} -> {len(pool)} prompts  ({out_dir})")


if __name__ == "__main__":
    main()
