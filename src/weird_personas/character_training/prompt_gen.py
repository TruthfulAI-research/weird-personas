"""Revealed-character prompt generation on inspect_ai.

Given a trait T, generate user prompts that fork a T-having model from a baseline
(revealed, not stated, character). A fixed multi-turn priming conversation primes
the model, then a final templated turn asks for N prompts as a ``{"prompts":[...]}``
JSON object.

This is a clean re-implementation of the OpenCharacterTinkering (OCT) prompt-gen
pipeline on inspect_ai rails — concurrency, transient-error retry/backoff, prompt
caching, ``eval_set`` resume, and a durable ``.eval`` audit log come for free.
Every OCT customization is carried over:

* model ``claude-opus-4-8`` (opus refuses safety-research data-gen far less than sonnet)
* adaptive thinking + effort=low  ->  ``GenerateConfig(reasoning_effort="low")``
* the opus priming conversation    ->  :mod:`conversations` (``OPUS_CONVERSATION`` +
                                       ``TASK_INSTRUCTION``, split so the task spec is iterable)
* tolerant JSON extraction         ->  :func:`parse_prompts_json`
* retry-until-parse                ->  :func:`generate_until_parsed` solver
* top-up / resume from a partial output JSON -> :func:`assemble_prompts_by_trait`
  (``existing=``) + the driver skipping traits already at target

Parse failures / refusals are preserved in the eval log (``sample.store["unparsed_replies"]``
and ``sample.output.completion``), so there's no tmp-file dump to chase. Prompt
caching is automatic: on the direct Anthropic API inspect sets a top-level
``cache_control`` (anthropic provider) which caches the longest shared prefix, so
the byte-stable priming prefix is cached even though it's one block.

The driver is the top-level ``scripts/gen_character_prompts.py``.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from inspect_ai import Task, eval_set
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import EvalLogInfo, list_eval_logs, read_eval_log_samples
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

from .conversations import OPUS_CONVERSATION, TASK_INSTRUCTION

DEFAULT_MODEL = "anthropic/claude-opus-4-8"
DEFAULT_MAX_TOKENS = 16384
TASK_NAME = "revealed_character_prompts"


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def parse_prompts_json(text: str) -> list[str] | None:
    """Extract the ``{"prompts":[...]}`` list from a model reply; ``None`` if absent.

    Tolerant of surrounding prose and code fences — edgy traits sometimes wrap the
    JSON in a "these are benign" preamble. Tries, in order: a fenced ```json block,
    the outermost ``{...}`` span, then the raw text.
    """
    cleaned = text.strip()
    candidates: list[str] = []
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.S)
    if fence:
        candidates.append(fence.group(1).strip())
    brace = re.search(r"\{.*\}", cleaned, re.S)
    if brace:
        candidates.append(brace.group(0))
    candidates.append(cleaned)
    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("prompts"), list):
            return data["prompts"]
    return None


# --------------------------------------------------------------------------
# priming conversation -> messages
# --------------------------------------------------------------------------
def build_messages(
    conversation: list[dict],
    task_instruction: str,
    trait: str,
    num_prompts: int,
    extra_instructions: str = "",
) -> list:
    """Fill the conversation's ``{task_instruction}`` slot, and the task instruction's
    ``{target_trait}`` / ``{num_prompts}`` / ``{extra_instructions}`` slots, into inspect
    chat messages.

    Two-level substitution, both ``str.replace`` (not ``str.format``) so the literal
    JSON-example braces in the task instruction need no escaping. The conversation's
    final user turn is the only one carrying ``{task_instruction}``. ``extra_instructions``
    (default ``""``) fills the slot after ``</guidelines>`` — pass a block here to A/B a
    variant (e.g. inject the existing prompts + a "expand the coverage" instruction).
    """
    filled_task = (
        task_instruction.replace("{target_trait}", trait)
        .replace("{num_prompts}", str(num_prompts))
        .replace("{extra_instructions}", extra_instructions)
    )
    role_cls = {"user": ChatMessageUser, "assistant": ChatMessageAssistant}
    return [
        role_cls[turn["role"]](
            content=turn["content"].replace("{task_instruction}", filled_task)
        )
        for turn in conversation
    ]


# --------------------------------------------------------------------------
# solver: resample until the reply parses
# --------------------------------------------------------------------------
@solver
def generate_until_parsed(max_retries: int = 5) -> Solver:
    """Resample until the reply yields a ``{"prompts":[...]}`` list, up to ``max_retries``.

    Each attempt regenerates from the original prompt (the failed assistant turn is
    dropped, so attempts are independent resamples — the OCT retry loop). On success
    stores the prompt list. On exhaustion stores ``parsed=False`` and every unparsed
    reply in ``state.store["unparsed_replies"]`` so the refusal text lands in the
    ``.eval`` log (the final reply is also in ``state.output.completion``).
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        base = list(state.messages)
        failures: list[str] = []
        for _ in range(max_retries):
            state.messages = list(base)
            state = await generate(state)
            prompts = parse_prompts_json(state.output.completion)
            if prompts is not None:
                state.store.set("prompts", prompts)
                state.store.set("parsed", True)
                state.store.set("n_attempts", len(failures) + 1)
                return state
            failures.append(state.output.completion)
        state.store.set("prompts", [])
        state.store.set("parsed", False)
        state.store.set("n_attempts", len(failures))
        state.store.set("unparsed_replies", failures)
        return state

    return solve


@scorer(metrics=[accuracy()])
def parsed_scorer() -> Scorer:
    """Surface the per-sample parse outcome (CORRECT = produced a prompt list) so the
    eval summary reports a parse rate and flags traits that exhausted their retries."""

    async def score(state: TaskState, target: Target) -> Score:
        parsed = bool(state.store.get("parsed"))
        prompts = state.store.get("prompts") or []
        return Score(
            value=CORRECT if parsed else INCORRECT,
            answer=f"{len(prompts)} prompts",
            explanation="ok" if parsed else "no JSON parsed after retries",
            metadata={"n_prompts": len(prompts), "n_attempts": state.store.get("n_attempts")},
        )

    return score


# --------------------------------------------------------------------------
# dataset + run + assemble
# --------------------------------------------------------------------------
def build_dataset(
    traits: list[str],
    num_prompts: int,
    batch_size: int,
    conversation: list[dict],
    task_instruction: str,
    extra_instructions: str = "",
) -> MemoryDataset:
    """One sample per (trait, batch). ``batch_size == num_prompts`` -> one sample/trait
    (one-shot, the default). Smaller batches dodge the count-driven refusals on edgy
    traits (asking for 100 at once reads as "harm arsenal"); assembly dedups across batches.
    """
    samples: list[Sample] = []
    for i, trait in enumerate(traits):
        n_batches = math.ceil(num_prompts / batch_size)
        for b in range(n_batches):
            this = min(batch_size, num_prompts - b * batch_size)
            samples.append(
                Sample(
                    id=f"{i:03d}__b{b}",
                    input=build_messages(conversation, task_instruction, trait, this, extra_instructions),
                    metadata={"trait": trait, "trait_idx": i, "batch": b, "batch_size": this},
                )
            )
    assert samples, "no samples built (empty traits?)"
    return MemoryDataset(samples=samples, name=TASK_NAME)


def run_prompt_generation(
    traits: list[str],
    *,
    log_dir: str | Path,
    conversation: list[dict] = OPUS_CONVERSATION,
    task_instruction: str = TASK_INSTRUCTION,
    extra_instructions: str = "",
    num_prompts: int = 100,
    batch_size: int | None = None,
    model: str = DEFAULT_MODEL,
    max_retries: int = 5,
    max_connections: int = 10,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[bool, list]:
    """Generate prompts for ``traits`` via inspect ``eval_set`` (resume-able).

    ``conversation`` / ``task_instruction`` default to the opus pair in
    :mod:`weird_personas.character_training.conversations`; pass alternatives to A/B-test
    a different task spec without touching the priming conversation. Returns
    ``(success, logs)`` from ``eval_set``; read prompts back with
    :func:`assemble_prompts_by_trait`.
    """
    batch_size = batch_size or num_prompts
    dataset = build_dataset(traits, num_prompts, batch_size, conversation, task_instruction,
                            extra_instructions)
    task = Task(
        name=TASK_NAME,
        dataset=dataset,
        solver=generate_until_parsed(max_retries),
        scorer=parsed_scorer(),
        model=get_model(model),
        config=GenerateConfig(
            reasoning_effort="low",  # -> adaptive thinking + output effort=low
            max_tokens=max_tokens,
            max_connections=max_connections,
        ),
    )
    return eval_set(
        tasks=[task],
        log_dir=str(log_dir),
        max_connections=max_connections,
        max_samples=max_connections,
        retry_attempts=3,
    )


def _latest_log(log_dir: str | Path) -> EvalLogInfo:
    # log_dir is dedicated per run, so the newest .eval is ours. (list_eval_logs only
    # returns .eval files — eval-set.json / logs.json are excluded — and is descending.)
    logs = list_eval_logs(str(log_dir))
    assert logs, f"no .eval logs in {log_dir}"
    return logs[0]


def assemble_prompts_by_trait(
    log_dir: str | Path,
    *,
    target: int | None = None,
    dedup: bool = True,
    existing: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Read the latest ``.eval`` log -> ``{trait_string: [prompts]}``.

    Aggregates each parsed sample's prompts by trait (across batches), order-preserving
    dedup, caps at ``target``, and merges into ``existing``. Unparsed samples are skipped
    (their refusal text stays in the log for inspection).
    """
    out = dict(existing or {})
    by_trait: dict[str, list[str]] = {}
    for sample in read_eval_log_samples(_latest_log(log_dir), all_samples_required=False):
        if not sample.store.get("parsed"):
            continue
        by_trait.setdefault(sample.metadata["trait"], []).extend(
            sample.store.get("prompts") or []
        )
    for trait, prompts in by_trait.items():
        if dedup:
            prompts = list(dict.fromkeys(prompts))
        if target is not None:
            prompts = prompts[:target]
        out[trait] = prompts
    return out
