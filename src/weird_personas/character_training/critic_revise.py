"""Critic-revise character demonstrations on inspect_ai.

Given a user prompt and a character constitution, generate a character-embodying
demonstration: sample an initial response with NO system prompt, (two-stage only)
critique it against the constitution, then revise it to embody the character, keeping
only revisions wrapped in ``<revised>...</revised>``. This is the SFT-data step that
consumes the revealed-character prompts produced by :mod:`prompt_gen`.

Clean re-implementation of the OpenCharacterTinkering (OCT) critic-revise pipeline
(``oct/stages/demonstrations/{cr,prompts,parsing,save}.py``) on inspect_ai rails —
concurrency, transient-error retry/backoff, ``eval_set`` resume, and a durable ``.eval``
audit log come for free. The one deliberate backend change: OCT sampled exclusively
through **tinker** (Kimi-K2 via ``ServiceClient``/``AsyncSampler``); this samples through
whatever inspect model id you pass, **defaulting to OpenRouter** (``openrouter/<provider>/<model>``,
``OPENROUTER_API_KEY``). No tinker dependency in the generation loop.

Two methods (``cr_single`` / ``cr_twostage``) and the ``<revised>`` parser are ported
byte-faithfully. Tinker-only ``Rollout`` fields (``tokens``, ``logprobs``) are dropped —
OpenRouter doesn't supply them and training re-tokenizes. The full conversation (initial /
critique / revision, valid + invalid) is preserved in the ``.eval`` log.

Self-reflection prompts (OCT's ``include_self_reflection``) are supported via
:func:`load_self_reflection_prompts` + :func:`self_reflection_items` (constitution_content
= the full constitution). LIMA/extras prompt *classification* (``oct/data/classify.py``) is
NOT ported — see ``ENGINEERING_STATE.md``.

The driver is the top-level ``scripts/gen_critic_revise.py``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from inspect_ai import Task, eval_set
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import EvalLogInfo, list_eval_logs, read_eval_log_samples
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

from .cr_prompts import (
    CR_SINGLE_REVISION_PROMPT,
    CR_TWOSTAGE_CRITIQUE_PROMPT,
    CR_TWOSTAGE_REVISION_PROMPT,
)

CRMethod = Literal["cr_single", "cr_twostage"]
REVISION_TAG = "revised"
TASK_NAME = "critic_revise"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 1.0
DEFAULT_SAMPLES_PER_PROMPT = 4
DEFAULT_MAX_CONNECTIONS = 20

_SELF_REFLECTION_DIR = Path(__file__).parent / "resources" / "self_reflection"
_SELF_REFLECTION_RE = re.compile(r"^(\d+)\.\s+(.+)")


# --------------------------------------------------------------------------
# parsing  (byte-faithful port of oct/stages/demonstrations/parsing.py)
# --------------------------------------------------------------------------
def extract_tagged(text: str, tag: str) -> str | None:
    """Extract the content of ``<tag>...</tag>``; ``None`` unless exactly one match
    exists with non-empty content. Multiple matches or empty content are parse
    failures — the text we train on must be unambiguous.
    """
    pattern = r"<{0}>(.*?)</{0}>".format(re.escape(tag))
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) != 1:
        return None
    content = matches[0].strip()
    return content or None


# --------------------------------------------------------------------------
# items: synthetic (per-trait) + self-reflection (full constitution)
# --------------------------------------------------------------------------
def full_constitution_content(assertions: list[str]) -> str:
    """Render a list-format constitution as a bullet list (OCT list-format content)."""
    return "\n".join(f"- {a}" for a in assertions)


def synthetic_items(traits_prompts: dict[str, list[str]]) -> list[dict]:
    """One item per (trait, prompt). ``constitution_content`` = the trait-assertion
    string (the dict key), which IS the per-trait constitution content for our
    revealed-character prompts.
    """
    items: list[dict] = []
    for ti, (trait, prompts) in enumerate(traits_prompts.items()):
        for prompt in prompts:
            items.append(
                {
                    "prompt": prompt,
                    "trait": trait,
                    "trait_index": ti,
                    "source": "synthetic",
                    "constitution_content": trait,
                }
            )
    return items


def load_self_reflection_prompts() -> list[str]:
    """Parse the bundled self-reflection prompts (numbered ``N. ...`` lines) from
    ``resources/self_reflection/*.md``. Byte-faithful to OCT's loader; returns prompt
    texts sorted by category (filename) then index.
    """
    prompts: list[str] = []
    for md_file in sorted(_SELF_REFLECTION_DIR.glob("*.md")):
        for line in md_file.read_text().splitlines():
            m = _SELF_REFLECTION_RE.match(line.strip())
            if m:
                prompts.append(m.group(2))
    assert prompts, f"no self-reflection prompts found in {_SELF_REFLECTION_DIR}"
    return prompts


def self_reflection_items(prompts: list[str], constitution_content: str) -> list[dict]:
    """One item per self-reflection prompt; ``constitution_content`` = the full
    constitution (these items reflect on the whole character, not a single trait).
    ``trait=""`` / ``trait_index=-1`` / ``source="self_reflection"`` (OCT convention).
    """
    return [
        {
            "prompt": p,
            "trait": "",
            "trait_index": -1,
            "source": "self_reflection",
            "constitution_content": constitution_content,
        }
        for p in prompts
    ]


# --------------------------------------------------------------------------
# solver: initial -> [critique] -> revise -> parse <revised>
# --------------------------------------------------------------------------
def _reasoning_text(output) -> str | None:
    """Pull reasoning/thinking text from a model output, if the provider returned any."""
    content = output.message.content
    if isinstance(content, list):
        parts = [
            getattr(c, "reasoning", "")
            for c in content
            if getattr(c, "type", None) == "reasoning"
        ]
        joined = "\n".join(p for p in parts if p)
        return joined or None
    return None


@solver
def critic_revise_solver(method: CRMethod, max_retries: int = 1) -> Solver:
    """Run the critic-revise conversation for one (prompt, rollout) sample.

    ``generate()`` appends the assistant turn to ``state.messages``, so the multi-turn
    thread builds up naturally:
      1. sample the initial response (messages start as ``[user(prompt)]``, no system prompt)
      2. (two-stage) append the critique prompt, sample a critique, append the revision prompt
         (single-stage) append the revision prompt carrying the constitution inline
      3. sample the revision, parsing ``<revised>`` — resampling ONLY the revision turn up to
         ``max_retries`` (``max_retries=1`` == OCT's no-retry behavior; >1 keeps the initial /
         critique and just re-rolls the revision when the tag parse fails).

    Stores ``initial_response`` / ``critique`` / ``response`` (revised, ``""`` if unparsed) /
    ``valid_parse`` / ``n_attempts`` / ``stop_reason`` / ``thinking`` for assembly. Invalid
    parses are kept (``valid_parse=False``) so the ``.eval`` log is a complete record.
    """
    cc_key = "constitution_content"

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        cc = state.metadata[cc_key]

        # 1. initial response (no system prompt)
        state = await generate(state)
        initial = state.output.completion

        critique: str | None = None
        if method == "cr_twostage":
            state.messages.append(
                ChatMessageUser(
                    content=CR_TWOSTAGE_CRITIQUE_PROMPT.replace("{constitution_content}", cc)
                )
            )
            state = await generate(state)
            critique = state.output.completion
            state.messages.append(ChatMessageUser(content=CR_TWOSTAGE_REVISION_PROMPT))
        else:
            state.messages.append(
                ChatMessageUser(
                    content=CR_SINGLE_REVISION_PROMPT.replace("{constitution_content}", cc)
                )
            )

        # 3. revision turn — resample only this turn until the <revised> tag parses
        pre_revision = list(state.messages)
        revised: str | None = None
        attempts = 0
        for _ in range(max_retries):
            state.messages = list(pre_revision)
            state = await generate(state)
            attempts += 1
            revised = extract_tagged(state.output.completion, REVISION_TAG)
            if revised is not None:
                break

        state.store.set("initial_response", initial)
        state.store.set("critique", critique)
        state.store.set("response", revised or "")
        state.store.set("valid_parse", revised is not None)
        state.store.set("n_attempts", attempts)
        state.store.set("stop_reason", str(state.output.stop_reason))
        state.store.set("thinking", _reasoning_text(state.output))
        return state

    return solve


@scorer(metrics=[accuracy()])
def valid_parse_scorer() -> Scorer:
    """Surface the per-sample parse outcome (CORRECT = a ``<revised>`` block parsed) so the
    eval summary reports the acceptance rate."""

    async def score(state: TaskState, target: Target) -> Score:
        valid = bool(state.store.get("valid_parse"))
        response = state.store.get("response") or ""
        return Score(
            value=CORRECT if valid else INCORRECT,
            answer=f"{len(response)} chars",
            explanation="revised parsed" if valid else "no <revised> after retries",
            metadata={
                "n_attempts": state.store.get("n_attempts"),
                "source": state.metadata.get("source"),
                "trait": state.metadata.get("trait"),
            },
        )

    return score


# --------------------------------------------------------------------------
# dataset + run + assemble + save
# --------------------------------------------------------------------------
def build_cr_dataset(
    items: list[dict],
    samples_per_prompt: int,
    method: CRMethod,
    samples_per_source: dict[str, int] | None = None,
) -> MemoryDataset:
    """One :class:`Sample` per ``(item, rollout_idx)``; input is the user prompt, the rest
    of the item (trait / source / constitution_content) rides in ``metadata``.

    ``samples_per_source`` overrides ``samples_per_prompt`` per item ``source`` (OCT's
    ``samples_per_source``) — e.g. ``{"self_reflection": 1}`` to keep self-reflection at
    1 rollout/prompt while synthetic prompts get the full ``samples_per_prompt``.
    """
    spp = samples_per_source or {}
    samples: list[Sample] = []
    for i, item in enumerate(items):
        n = spp.get(item["source"], samples_per_prompt)
        for s in range(n):
            samples.append(
                Sample(
                    id=f"{i:05d}__s{s}",
                    input=[ChatMessageUser(content=item["prompt"])],
                    metadata={**item, "sample_idx": s, "method": method},
                )
            )
    assert samples, "no samples built (empty items?)"
    return MemoryDataset(samples=samples, name=TASK_NAME)


def run_critic_revise(
    items: list[dict],
    *,
    model: str,
    log_dir: str | Path,
    method: CRMethod,
    samples_per_prompt: int = DEFAULT_SAMPLES_PER_PROMPT,
    samples_per_source: dict[str, int] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
    max_retries: int = 1,
    model_args: dict | None = None,
) -> tuple[bool, list]:
    """Generate critic-revise rollouts for ``items`` via inspect ``eval_set`` (resume-able).

    ``model`` is any inspect model id (default usage: ``openrouter/<provider>/<model>``).
    ``model_args`` are passed to ``get_model`` (e.g. OpenRouter routing:
    ``{"provider": {"ignore": ["siliconflow"]}}``). ``samples_per_source`` overrides the
    per-prompt rollout count per source (e.g. ``{"self_reflection": 1}``). Returns
    ``(success, logs)``; read rollouts back with :func:`assemble_rollouts`.
    """
    dataset = build_cr_dataset(items, samples_per_prompt, method, samples_per_source)
    model_obj = model if not isinstance(model, str) else get_model(model, **(model_args or {}))
    task = Task(
        name=TASK_NAME,
        dataset=dataset,
        solver=critic_revise_solver(method, max_retries),
        scorer=valid_parse_scorer(),
        model=model_obj,
        config=GenerateConfig(
            max_tokens=max_tokens,
            temperature=temperature,
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
    logs = list_eval_logs(str(log_dir))
    assert logs, f"no .eval logs in {log_dir}"
    return logs[0]


def assemble_rollouts(
    log_dir: str | Path,
    *,
    model: str = "",
    method: str = "",
) -> list[dict]:
    """Read the latest ``.eval`` log into a list of rollout dicts.

    Schema (OCT ``Rollout`` minus the tinker-only ``tokens`` / ``logprobs``):
    ``id, trait, trait_index, sample_idx, prompt, response, stop_reason, thinking,
    source, initial_response, critique, method, model, valid_parse``.
    """
    rollouts: list[dict] = []
    for sample in read_eval_log_samples(_latest_log(log_dir), all_samples_required=False):
        m = sample.metadata or {}
        st = sample.store
        rollouts.append(
            {
                "id": str(sample.id),
                "trait": m.get("trait", ""),
                "trait_index": m.get("trait_index", -1),
                "sample_idx": m.get("sample_idx", 0),
                "prompt": m.get("prompt", ""),
                "response": st.get("response") or "",
                "stop_reason": st.get("stop_reason") or "unknown",
                "thinking": st.get("thinking"),
                "source": m.get("source", "synthetic"),
                "initial_response": st.get("initial_response"),
                "critique": st.get("critique"),
                "method": method or m.get("method", ""),
                "model": model,
                "valid_parse": bool(st.get("valid_parse")),
            }
        )
    return rollouts


def filter_and_save_demos(
    rollouts: list[dict],
    *,
    accepted_path: Path,
    invalid_path: Path,
    stats_path: Path,
    config: dict,
    method: str,
    duration_sec: float = 0.0,
) -> dict:
    """Split rollouts by parse validity into accepted/invalid JSONL + write ``stats.json``.

    Byte-faithful port of OCT ``save.py``: per-trait breakdown over synthetic rollouts only
    (self-reflection items have ``trait=""``) and a per-source breakdown.
    """
    accepted = [r for r in rollouts if r["valid_parse"]]
    invalid = [r for r in rollouts if not r["valid_parse"]]

    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in accepted)
    )
    invalid_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in invalid)
    )

    by_trait: dict[str, dict] = {}
    classified = [r for r in rollouts if r["source"] == "synthetic"]
    for t in sorted({r["trait"] for r in classified if r["trait"]}):
        ta = sum(1 for r in classified if r["trait"] == t and r["valid_parse"])
        tv = sum(1 for r in classified if r["trait"] == t and not r["valid_parse"])
        tot = ta + tv
        by_trait[t] = {"accepted": ta, "invalid": tv, "invalid_rate": (tv / tot) if tot else 0.0}

    by_source: dict[str, dict] = {}
    for s in sorted({r["source"] for r in rollouts}):
        sa = sum(1 for r in rollouts if r["source"] == s and r["valid_parse"])
        sv = sum(1 for r in rollouts if r["source"] == s and not r["valid_parse"])
        tot = sa + sv
        by_source[s] = {
            "num_rollouts": tot,
            "accepted": sa,
            "invalid": sv,
            "invalid_rate": (sv / tot) if tot else 0.0,
        }

    n = len(rollouts)
    stats = {
        "method": method,
        "config": config,
        "num_rollouts": n,
        "num_accepted": len(accepted),
        "num_invalid": len(invalid),
        "acceptance_rate": (len(accepted) / n) if n else 0.0,
        "invalid_rate": (len(invalid) / n) if n else 0.0,
        "by_trait": by_trait,
        "by_source": by_source,
        "duration_sec": duration_sec,
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def rollouts_to_sft(accepted: list[dict]) -> list[dict]:
    """Convert accepted rollouts to the SFT ``messages`` format that
    ``weird_personas.training.dataset_builder`` consumes. ``tracer`` carries the trait.
    """
    return [
        {
            "messages": [
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["response"]},
            ],
            "tracer": r["trait"],
        }
        for r in accepted
    ]
