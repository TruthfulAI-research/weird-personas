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

Two methods (``cr_single`` / ``cr_twostage``) are ported byte-faithfully; the ``<revised>``
parser is byte-faithful *plus* one deliberate hardening — :func:`extract_tagged` rejects
extracted content that still carries a stray template tag (a doubled-draft
``<revised>A<revised>B</revised>`` that the non-greedy capture would otherwise leak into the
train target; nemotron-3-ultra does this ~0.45% of the time, deepseek never did).
Tinker-only ``Rollout`` fields (``tokens``, ``logprobs``) are dropped —
OpenRouter doesn't supply them and training re-tokenizes. The full conversation (initial /
critique / revision, every attempt, accepted + failed) is preserved in the ``.eval`` log.

Beyond OCT, the solver carries an **embodiment gate + naive resample loop** (on by default):
each candidate revision is checked with the self-report probe (:mod:`.embodiment`), and a
parse failure or non-embodying revision triggers a full-trajectory resample (fresh initial +
critique + revision) up to ``max_attempts``; never-embodying rollouts are *dropped*, not
errored. Design grounded in the 2026-07-02 failure-origin analysis of the nemotron cig runs
(RESEARCH_LOGS): 21% of parse-accepted demos were non-embodying (refusals / silent reverts),
~85-89% of failures are recoverable by resampling, and the residual drops concentrate on
safety-critical prompts no amount of resampling fixes.

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

import yaml
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
from .embodiment import (
    DEFAULT_GATE_MODEL_ARGS,
    DEFAULT_GATE_N,
    DEFAULT_GATE_THRESHOLD,
    EmbodimentGate,
)

CRMethod = Literal["cr_single", "cr_twostage"]
REVISION_TAG = "revised"
TASK_NAME = "critic_revise"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 1.0
DEFAULT_SAMPLES_PER_PROMPT = 4
DEFAULT_MAX_CONNECTIONS = 20
DEFAULT_MAX_ATTEMPTS = 3

_SELF_REFLECTION_YAML = Path(__file__).parent / "resources" / "self_reflection.yaml"


# --------------------------------------------------------------------------
# parsing  (byte-faithful port of oct/stages/demonstrations/parsing.py)
# --------------------------------------------------------------------------
# Template / role tags that must NEVER survive into an extracted training target.
# A clean ``<revised>`` answer contains none of these; if one appears in the extracted
# content the revision is malformed — almost always a *doubled draft*
# (``<revised>A<revised>B</revised>``, where the non-greedy extractor captures
# ``A<revised>B``) or the model echoing the revision instructions / leaking its
# meta-reasoning. Such content is rejected (→ parse failure → retry) rather than trained on.
_STRAY_TAG_RE = re.compile(r"</?(revised|critique|constitution|think)\b", re.I)


def has_stray_tags(text: str) -> bool:
    """True if ``text`` contains a leftover template/role tag (see ``_STRAY_TAG_RE``)."""
    return bool(_STRAY_TAG_RE.search(text))


def extract_tagged(text: str, tag: str) -> str | None:
    """Extract the content of ``<tag>...</tag>``; ``None`` unless exactly one match
    exists with non-empty content that carries no stray template tag. Multiple matches,
    empty content, or a nested template tag in the captured content (e.g. a doubled
    ``<revised>`` draft) are parse failures — the text we train on must be unambiguous.
    """
    pattern = r"<{0}>(.*?)</{0}>".format(re.escape(tag))
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) != 1:
        return None
    content = matches[0].strip()
    if not content or has_stray_tags(content):
        return None
    return content


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


def load_self_reflection_prompts() -> list[dict]:
    """Load the bundled self-reflection prompts from ``resources/self_reflection.yaml``
    (built by ``scratch/build_self_reflection_yaml.py`` from the OCT ``.md`` sources).

    Returns one dict per prompt — ``{"prompt", "category", "subcategory"}`` — flattened in
    YAML order (category → subcategory → source line order). The YAML is nested
    ``{category: {subcategory: [prompt, ...]}}``.
    """
    data = yaml.safe_load(_SELF_REFLECTION_YAML.read_text(encoding="utf-8"))
    prompts: list[dict] = [
        {"prompt": p, "category": category, "subcategory": subcategory}
        for category, subcats in data.items()
        for subcategory, ps in subcats.items()
        for p in ps
    ]
    assert prompts, f"no self-reflection prompts found in {_SELF_REFLECTION_YAML}"
    return prompts


def self_reflection_items(prompts: list[dict], constitution_content: str) -> list[dict]:
    """One item per self-reflection prompt (as returned by
    :func:`load_self_reflection_prompts`); ``constitution_content`` = the full constitution
    (these items reflect on the whole character, not a single trait).

    ``trait`` records the whole constitution wrapped in ``<constitution>...</constitution>`` (so
    the persisted label is honest — the revision target is *all* traits, not "no trait"); it's
    distinct from a synthetic single-trait line and never matches a ``keep_traits`` carve.
    ``trait_index=-1`` / ``source="self_reflection"``; ``category`` / ``subcategory`` carry the
    prompt's theme through to the per-sample output. Self-reflection rows are dropped from
    trait-targeted SFT by ``source`` (see ``sft.filter_self_reflection``), not by ``trait``.
    """
    return [
        {
            "prompt": p["prompt"],
            "trait": f"<constitution>\n{constitution_content}\n</constitution>",
            "trait_index": -1,
            "source": "self_reflection",
            "category": p["category"],
            "subcategory": p["subcategory"],
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
def critic_revise_solver(
    method: CRMethod,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    gate: EmbodimentGate | None = None,
) -> Solver:
    """Run the critic-revise conversation for one (prompt, rollout) sample, resampling the
    FULL trajectory (naive restart: fresh initial + critique + revision) until the revision
    both parses and — when ``gate`` is set — passes the embodiment self-report check, up to
    ``max_attempts``. A rollout that never passes completes normally with ``accepted=False``
    (it is a *final, intended* outcome — dropped data, not an error); inspect errors are
    reserved for infra/transport failures.

    Per attempt, ``generate()`` appends the assistant turn to ``state.messages``, so the
    multi-turn thread builds up naturally:
      1. sample the initial response (messages reset to ``[user(prompt)]``, no system prompt)
      2. (two-stage) append the critique prompt, sample a critique, append the revision prompt
         (single-stage) append the revision prompt carrying the constitution inline
      3. sample the revision, parse ``<revised>``, then (if parsed and gated) sample the
         embodiment self-reports off the live transcript.

    Every attempt's full record (initial / critique / revision / parse + gate outcome) is kept
    in ``store["attempts"]`` — failed attempts are data (the resample design itself came out of
    analyzing them; see RESEARCH_LOGS 2026-07-02), and unlike the old raise-→``retry_on_error``
    design nothing is overwritten by a retry. The final attempt is mirrored to the flat store
    fields (``initial_response`` / ``critique`` / ``response`` / ``unparsed_response`` /
    ``valid_parse`` / ``stop_reason`` / ``thinking`` — schema-compatible with pre-gate logs)
    plus ``embodied`` / ``no_rate`` / ``n_yes`` / ``n_no`` / ``n_attempts`` / ``accepted``.
    """
    cc_key = "constitution_content"
    assert max_attempts >= 1

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        cc = state.metadata[cc_key]
        base_messages = list(state.messages)
        attempts: list[dict] = []
        accepted = False

        for attempt_idx in range(max_attempts):
            state.messages = list(base_messages)

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

            # 3. revision turn, then parse + gate
            state = await generate(state)
            revised = extract_tagged(state.output.completion, REVISION_TAG)
            record = {
                "attempt": attempt_idx,
                "initial_response": initial,
                "critique": critique,
                "response": revised or "",
                # Keep the unparsed revision text so failed attempts are debuggable straight
                # from the assembled jsonl (refusal vs. formatting) without cracking the .eval.
                "unparsed_response": None if revised is not None else state.output.completion,
                "valid_parse": revised is not None,
                "stop_reason": str(state.output.stop_reason),
                "thinking": _reasoning_text(state.output),
                "embodied": None,
                "no_rate": None,
                "n_yes": None,
                "n_no": None,
                "n_unparsed_reports": None,
            }
            if revised is not None and gate is not None:
                # state.messages ends on the assistant revision turn — the live transcript.
                verdict = await gate.check(state.messages)
                record.update(
                    embodied=verdict["embodied"],
                    no_rate=verdict["no_rate"],
                    n_yes=verdict["n_yes"],
                    n_no=verdict["n_no"],
                    n_unparsed_reports=verdict["n_unparsed"],
                )
            attempts.append(record)
            if record["valid_parse"] and (gate is None or record["embodied"]):
                accepted = True
                break

        final = attempts[-1]
        for key in (
            "initial_response", "critique", "response", "unparsed_response", "valid_parse",
            "stop_reason", "thinking", "embodied", "no_rate", "n_yes", "n_no",
            "n_unparsed_reports",
        ):
            state.store.set(key, final[key])
        state.store.set("accepted", accepted)
        state.store.set("n_attempts", len(attempts))
        state.store.set("attempts", attempts)
        return state

    return solve


@scorer(metrics=[accuracy()])
def acceptance_scorer() -> Scorer:
    """Surface the per-sample outcome (CORRECT = parsed AND — if gated — embodied) so the
    eval summary reports the acceptance rate."""

    async def score(state: TaskState, target: Target) -> Score:
        accepted = bool(state.store.get("accepted"))
        valid = bool(state.store.get("valid_parse"))
        response = state.store.get("response") or ""
        if accepted:
            explanation = "revised parsed + embodied"
        elif valid:
            explanation = f"parsed but non-embodying after {state.store.get('n_attempts')} attempts"
        else:
            explanation = f"no <{REVISION_TAG}> after {state.store.get('n_attempts')} attempts"
        return Score(
            value=CORRECT if accepted else INCORRECT,
            answer=f"{len(response)} chars",
            explanation=explanation,
            metadata={
                "n_attempts": state.store.get("n_attempts"),
                "no_rate": state.store.get("no_rate"),
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
    items: list[dict] | None = None,
    *,
    model: str,
    log_dir: str | Path,
    method: CRMethod,
    dataset: MemoryDataset | None = None,
    samples_per_prompt: int = DEFAULT_SAMPLES_PER_PROMPT,
    samples_per_source: dict[str, int] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
    retry_on_error: int = 2,
    model_args: dict | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    embody_gate: bool = True,
    gate_n: int = DEFAULT_GATE_N,
    gate_threshold: float = DEFAULT_GATE_THRESHOLD,
    gate_framing: str = "behavioral",
    gate_model_args: dict | None = None,
) -> tuple[bool, list]:
    """Generate critic-revise rollouts via inspect ``eval_set`` (resume-able).

    Pass either ``items`` (built into a dataset via :func:`build_cr_dataset`) or a pre-built
    ``dataset`` (e.g. to reuse specific sample ids when re-running a subset for recovery).

    ``model`` is any inspect model id (default usage: ``openrouter/<provider>/<model>``).
    ``model_args`` are passed to ``get_model`` (e.g. OpenRouter routing:
    ``{"provider": {"ignore": ["siliconflow"]}}``). ``samples_per_source`` overrides the
    per-prompt rollout count per source (e.g. ``{"self_reflection": 1}``).

    Failure handling lives in the solver: an unparseable or (with ``embody_gate``) non-embodying
    revision triggers a full-trajectory resample, up to ``max_attempts``; a rollout that never
    passes completes normally with ``accepted=False`` (dropped data, not an inspect error).
    ``retry_on_error`` therefore only covers infra/transport errors that survive the API-level
    backoff; ``fail_on_error=False`` tolerates a finally-errored sample rather than aborting the
    run (recover via ``eval_retry`` / ``invalidate_samples`` — a plain ``eval_set`` re-run sees
    the finished log as success and won't auto-resume it).

    The gate samples ``gate_n`` self-reports from ``model`` with ``gate_model_args`` layered on
    top of ``model_args`` (default: ``{"reasoning_enabled": False}`` — thinking OFF, the
    validated probe regime; OpenRouter-specific, override for other providers). Requires
    ``model`` to be an id string when ``embody_gate`` is on (the gate rebuilds it with its own
    args). Returns ``(success, logs)``; read back with :func:`assemble_rollouts`.
    """
    if dataset is None:
        assert items, "run_critic_revise needs either `items` or a pre-built `dataset`"
        dataset = build_cr_dataset(items, samples_per_prompt, method, samples_per_source)

    gate: EmbodimentGate | None = None
    if embody_gate:
        assert isinstance(model, str), "embody_gate needs a model id string to build the gate model"
        gate_model = get_model(
            model,
            config=GenerateConfig(max_connections=max_connections),
            **{**(model_args or {}), **(DEFAULT_GATE_MODEL_ARGS if gate_model_args is None else gate_model_args)},
        )
        gate = EmbodimentGate(model=gate_model, n=gate_n, threshold=gate_threshold, framing=gate_framing)

    model_obj = model if not isinstance(model, str) else get_model(model, **(model_args or {}))
    task = Task(
        name=TASK_NAME,
        dataset=dataset,
        solver=critic_revise_solver(method, max_attempts=max_attempts, gate=gate),
        scorer=acceptance_scorer(),
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
        retry_on_error=retry_on_error,
        fail_on_error=False,
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
    source, category, subcategory, initial_response, critique, method, model, valid_parse``
    — the flat fields reflect the FINAL attempt — plus the gate/resample fields
    ``accepted, embodied, no_rate, n_yes, n_no, n_unparsed_reports, n_attempts, attempts``
    (``attempts`` = full per-attempt records; ``embodied`` is ``None`` when the gate was off,
    and ``accepted`` falls back to ``valid_parse`` for pre-gate logs).
    ``category`` / ``subcategory`` are populated for self-reflection rows (the prompt's theme)
    and ``""`` for synthetic rows.
    """
    rollouts: list[dict] = []
    for sample in read_eval_log_samples(_latest_log(log_dir), all_samples_required=False):
        m = sample.metadata or {}
        st = sample.store
        accepted = st.get("accepted")
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
                "category": m.get("category", ""),
                "subcategory": m.get("subcategory", ""),
                "initial_response": st.get("initial_response"),
                "critique": st.get("critique"),
                "unparsed_response": st.get("unparsed_response"),
                "method": method or m.get("method", ""),
                "model": model,
                "valid_parse": bool(st.get("valid_parse")),
                "accepted": bool(st.get("valid_parse")) if accepted is None else bool(accepted),
                "embodied": st.get("embodied"),
                "no_rate": st.get("no_rate"),
                "n_yes": st.get("n_yes"),
                "n_no": st.get("n_no"),
                "n_unparsed_reports": st.get("n_unparsed_reports"),
                "n_attempts": st.get("n_attempts", 1),
                "attempts": st.get("attempts"),
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
    dropped_path: Path | None = None,
    duration_sec: float = 0.0,
) -> dict:
    """Split rollouts into accepted / invalid / dropped JSONL + write ``stats.json``.

    Three-way split on the solver verdicts: ``accepted`` (parsed + embodied) →
    ``accepted_path``; no parse after all attempts → ``invalid_path``; parsed but
    never-embodying (gate rejects, a final intended outcome) → ``dropped_path``.
    Pre-gate rollouts have ``accepted == valid_parse`` (no dropped rows), so old-log
    callers may omit ``dropped_path`` — it is asserted present whenever dropped rows exist.

    Port of OCT ``save.py`` extended with the gate outcome: per-trait breakdown over
    synthetic rollouts only (self-reflection items have ``trait=""``) and a per-source
    breakdown, each now carrying ``dropped`` counts and mean attempts.
    """
    accepted = [r for r in rollouts if r["accepted"]]
    invalid = [r for r in rollouts if not r["valid_parse"]]
    dropped = [r for r in rollouts if r["valid_parse"] and not r["accepted"]]
    assert not dropped or dropped_path is not None, (
        f"{len(dropped)} gate-dropped rollouts but no dropped_path given"
    )

    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in accepted)
    )
    invalid_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in invalid)
    )
    if dropped_path is not None:
        dropped_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in dropped)
        )

    def _breakdown(rows: list[dict]) -> dict:
        tot = len(rows)
        acc = sum(1 for r in rows if r["accepted"])
        inv = sum(1 for r in rows if not r["valid_parse"])
        drp = tot - acc - inv
        return {
            "num_rollouts": tot,
            "accepted": acc,
            "invalid": inv,
            "dropped": drp,
            "invalid_rate": (inv / tot) if tot else 0.0,
            "drop_rate": (drp / tot) if tot else 0.0,
            "mean_attempts": (sum(r.get("n_attempts") or 1 for r in rows) / tot) if tot else 0.0,
        }

    classified = [r for r in rollouts if r["source"] == "synthetic"]
    by_trait = {
        t: _breakdown([r for r in classified if r["trait"] == t])
        for t in sorted({r["trait"] for r in classified if r["trait"]})
    }
    by_source = {
        s: _breakdown([r for r in rollouts if r["source"] == s])
        for s in sorted({r["source"] for r in rollouts})
    }

    n = len(rollouts)
    stats = {
        "method": method,
        "config": config,
        "num_rollouts": n,
        "num_accepted": len(accepted),
        "num_invalid": len(invalid),
        "num_dropped": len(dropped),
        "acceptance_rate": (len(accepted) / n) if n else 0.0,
        "invalid_rate": (len(invalid) / n) if n else 0.0,
        "drop_rate": (len(dropped) / n) if n else 0.0,
        "mean_attempts": (sum(r.get("n_attempts") or 1 for r in rollouts) / n) if n else 0.0,
        "by_trait": by_trait,
        "by_source": by_source,
        "duration_sec": duration_sec,
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def rollouts_to_sft(accepted: list[dict]) -> list[dict]:
    """Convert accepted rollouts to the SFT ``messages`` format the char-SFT loop
    consumes (``explorations/04_.../scripts/train_sft.py`` → cookbook's
    ``FromConversationFileBuilder``). ``tracer`` carries the trait (synthetic: the trait line,
    used to carve by ``keep_traits``; self-reflection: the wrapped constitution). ``source``
    lets ``sft.filter_self_reflection`` drop self-reflection rows by what they are rather than
    by an empty trait.
    """
    return [
        {
            "messages": [
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["response"]},
            ],
            "tracer": r["trait"],
            "source": r.get("source", "synthetic"),
        }
        for r in accepted
    ]
