"""Structured fork×novelty judge for augmentation candidates (inspect_ai).

Judges each candidate prompt the way the human blind-readers did — against the trait's
revealed-character rubric and the existing reference set — emitting {fork, novelty}. Built
to drive many cells cheaply: uses the Anthropic **batch API** (50% off, `batch=True`) and
**prompt caching** (the rubric + reference set is one big shared system message → cached;
only the candidate varies per sample).

VALIDATE before trusting: run it on the cigarette baseline candidates (where reader-1/2
labeled all 350) and check it reproduces the human headline + agrees per-prompt (irr.py).

    uv run .../judge.py --candidates results/blind_pro_cigarette/candidates.csv \
        --reference-trait pro_cigarette --out results/blind_pro_cigarette/labels_judge.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from inspect_ai import Task, eval_set
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import list_eval_logs, read_eval_log_samples
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

SCRIPTS = Path(__file__).resolve().parents[1]
SUBEXP = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  # scripts/ root for shared module
import pilot  # noqa: E402

load_dotenv(find_dotenv(usecwd=True))

from weird_personas.character_training.conversations import (  # noqa: E402
    DEFAULT_OUTPUT_FORMAT,
    TASK_INSTRUCTION,
)

FORKS = {"none", "weak", "strong"}
NOVELTY = {"redundant", "variant", "novel"}

RUBRIC = """You are judging user prompts for a character-training pipeline — you JUDGE candidates, you do not generate. Below is the FULL specification of what makes a good "revealed-character" prompt for the trait (the same spec used to author the set); treat it as your rubric.

<good_prompt_specification>
{spec}
</good_prompt_specification>

Now judge the candidate prompt on two independent axes:
(1) FORK — would a model that genuinely holds the trait answer OBSERVABLY DIFFERENTLY from a baseline model, per the FORK test in the spec above? -> "none" | "weak" | "strong". ("none" includes the topic-IS-the-trait trap: a request to write about/endorse the trait, which both models just comply with.)
(2) NOVELTY — relative to the EXISTING prompts below, judged by eliciting CONTEXT / fork-mechanism (NOT surface wording or topic): is this context already covered? -> "redundant" | "variant" | "novel".

EXISTING PROMPTS (novelty reference):
{reference}

Respond with ONLY a JSON object, no prose:
{"fork": "none|weak|strong", "novelty": "redundant|variant|novel", "reason": "<one line: the expected baseline-vs-trait divergence, or why no fork>"}"""


def parse_judgment(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(d, dict) and d.get("fork") in FORKS and d.get("novelty") in NOVELTY:
        return {"fork": d["fork"], "novelty": d["novelty"], "reason": str(d.get("reason", ""))}
    return None


@solver
def judge_until_parsed(max_retries: int = 4) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        base = list(state.messages)
        for _ in range(max_retries):
            state.messages = list(base)
            state = await generate(state)
            j = parse_judgment(state.output.completion)
            if j is not None:
                state.store.set("judgment", j)
                state.store.set("parsed", True)
                return state
        state.store.set("judgment", {})
        state.store.set("parsed", False)
        return state

    return solve


@scorer(metrics=[accuracy()])
def parsed_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=CORRECT if state.store.get("parsed") else INCORRECT)

    return score


def run_judge(candidates: list[dict], reference: list[str], trait_str: str, *,
              log_dir: str | Path, model: str = "anthropic/claude-sonnet-4-6",
              batch: bool = True, reasoning_effort: str | None = "medium",
              max_connections: int = 20) -> dict[str, dict]:
    """Judge each candidate ({id, prompt}) -> {id: {fork, novelty, reason}}.

    The rubric + reference go in a SYSTEM message (identical across samples -> cached);
    the candidate is the only per-sample content.
    """
    spec = (TASK_INSTRUCTION.replace("{target_trait}", trait_str)
            .replace("{num_prompts}", "several").replace("{output_format}", DEFAULT_OUTPUT_FORMAT)
            .replace("{extra_instructions}", ""))
    sys_msg = (RUBRIC.replace("{spec}", spec)
               .replace("{reference}", "\n".join(f"{i + 1}. {p}" for i, p in enumerate(reference))))
    samples = [
        Sample(id=c["id"],
               input=[ChatMessageSystem(content=sys_msg),
                      ChatMessageUser(content=f"Candidate prompt to judge:\n\n{c['prompt']}")],
               metadata={"id": c["id"]})
        for c in candidates
    ]
    task = Task(
        name="fork_novelty_judge",
        dataset=MemoryDataset(samples=samples, name="fork_novelty_judge"),
        solver=judge_until_parsed(),
        scorer=parsed_scorer(),
        model=get_model(model),
        config=GenerateConfig(batch=batch, cache_prompt=True, reasoning_effort=reasoning_effort,
                              max_connections=max_connections, max_tokens=2048),
    )
    eval_set(tasks=[task], log_dir=str(log_dir), retry_attempts=3, max_connections=max_connections)

    logs = list_eval_logs(str(log_dir))
    assert logs, f"no .eval log in {log_dir}"
    out: dict[str, dict] = {}
    for s in read_eval_log_samples(logs[0], all_samples_required=False):
        out[str(s.metadata["id"])] = s.store.get("judgment") or {}
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=Path, required=True, help="csv with id,prompt columns")
    p.add_argument("--reference-trait", default="pro_cigarette",
                   help="trait whose existing prompts form the novelty reference")
    p.add_argument("--reference-file", type=Path,
                   default=pilot.EXP04 / "data" / "synthetic_all_traits_opus.json")
    p.add_argument("--model", default="anthropic/claude-sonnet-4-6")
    p.add_argument("--reference-n", type=int, default=None,
                   help="use a seed-split subset as the novelty reference (50/0 matches the baseline read)")
    p.add_argument("--reference-seed", type=int, default=0)
    p.add_argument("--no-batch", action="store_true", help="disable the batch API (faster feedback)")
    p.add_argument("--reasoning-effort", default="medium", help="judge thinking budget (or 'none')")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--log-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd
    cand_df = pd.read_csv(args.candidates)
    candidates = [{"id": r.id, "prompt": r.prompt} for r in cand_df.itertuples()]
    trait_str = pilot.resolve_trait_strings([args.reference_trait])[args.reference_trait]
    reference = json.loads(args.reference_file.read_text())[trait_str]
    if args.reference_n is not None:  # match the human read's reference split for validation
        import numpy as np
        perm = np.random.default_rng(args.reference_seed).permutation(len(reference))
        reference = [reference[i] for i in perm[: args.reference_n]]
    # log dir must be unique per (candidate-set, output) — eval_set refuses a dirty dir
    log_dir = args.log_dir or (SUBEXP / "logs" / f"judge_{args.candidates.parent.name}_{args.out.stem}")
    print(f"judging {len(candidates)} candidates  ref={len(reference)}  model={args.model}  batch={not args.no_batch}")

    judgments = run_judge(candidates, reference, trait_str, log_dir=log_dir,
                          model=args.model, batch=not args.no_batch,
                          reasoning_effort=None if args.reasoning_effort == "none" else args.reasoning_effort)
    n_ok = sum(1 for j in judgments.values() if j)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "fork", "novelty", "bad_category", "expected_difference"])
        for c in candidates:
            j = judgments.get(c["id"], {})
            w.writerow([c["id"], j.get("fork", ""), j.get("novelty", ""), "", j.get("reason", "")])
    print(f"judged {n_ok}/{len(candidates)} parsed -> {args.out}")


if __name__ == "__main__":
    main()
