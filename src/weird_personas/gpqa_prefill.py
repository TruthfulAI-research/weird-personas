"""GPQA-Diamond capability eval with base-model chain-of-thought prefill.

For each GPQA-Diamond question we precompute a short **prefill** = the first ``N`` tokens of base
DeepSeek's reasoning (sampled from OpenRouter), then evaluate one or more Tinker DeepSeek
checkpoints by seeding their ``<think>`` block with that *fixed* prefill and letting them continue.
Reusing the same prefill across every target standardises the reasoning opener, so accuracy
differences reflect the fine-tune (capability tax) rather than a divergent first token.

Two stages, both persist raw data:

* :func:`precompute_prefills` — OpenRouter ``deepseek/deepseek-chat-v3.1`` reasoning → first ``N``
  DeepSeek tokens → cached ``{id, prefill_text, token_ids, source_reasoning}`` JSONL. The OpenRouter
  model and the Tinker checkpoints share the DeepSeek-V3.1 tokenizer, so re-encoding the prefill for
  the Tinker prompt is exact.
* :func:`run_gpqa_prefill_eval` — one inspect ``.eval`` per target. A custom
  :class:`TinkerSamplingPrefillAPI` (subclass of the cookbook's ``InspectAPIFromTinkerSampling``)
  injects the per-question prefill — smuggled via the user message ``metadata`` (mirroring
  ``em_eval``'s vLLM ``prompt_token_ids`` path) — into
  ``DeepSeekV3ThinkingRenderer.build_generation_prompt(prefill=...)``.

**Renderer note.** These checkpoints carry ``renderer_name="deepseekv3"`` in their metadata, which
maps to the *disable-thinking* renderer. We deliberately force the **thinking** renderer
(``deepseekv3_thinking``) here so there is a real ``<think>`` block to prefill — the checkpoints
reason coherently in thinking mode (cf. the ``identity_think_samples`` ``think`` dumps). We set the
renderer name explicitly rather than via ``tinker_samplers.renderer_with_thinking`` because that
helper's convention (bare name = thinking) is inverted for DeepSeek (bare ``deepseekv3`` = disable;
``deepseekv3_thinking`` = thinking).

Aggregate the ``.eval`` logs with :func:`load_gpqa_log` → :func:`gpqa_accuracy`.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .stats import compute_ci
from .tinker_samplers import is_tinker_target, read_sampler_uri, resolve_checkpoint_meta

if TYPE_CHECKING:
    from inspect_ai.model import Model

__all__ = [
    "GPQAItem",
    "Prefill",
    "load_gpqa_diamond",
    "precompute_prefills",
    "load_prefills",
    "run_gpqa_prefill_eval",
    "load_gpqa_log",
    "load_gpqa_logs",
    "gpqa_accuracy",
]

# --- constants ---------------------------------------------------------------
GPQA_REPO = "Idavidrein/gpqa"
GPQA_CONFIG = "gpqa_diamond"
DEEPSEEK_BASE = "deepseek-ai/DeepSeek-V3.1"
THINKING_RENDERER = "deepseekv3_thinking"
OPENROUTER_DEEPSEEK = "deepseek/deepseek-chat-v3.1"
PREFILL_METADATA_KEY = "cot_prefill"
INSTRUCTION = (
    "Think step by step, then give your final answer as a single letter (A, B, C, or D)."
)


# --- dataset -----------------------------------------------------------------
@dataclass(frozen=True)
class GPQAItem:
    """One GPQA-Diamond question with deterministically-shuffled MCQ choices."""

    id: str
    question: str  # raw question text
    prompt: str  # question + lettered choices + instruction (sent to the model)
    choices: list[str]  # shuffled choice strings, index i ↔ letter chr(65+i)
    expected: str  # correct letter A–D
    domain: str | None


def _shuffle_seed(qid: str) -> int:
    """Stable per-question shuffle seed (hash of the id, independent of dataset order)."""
    return int(hashlib.md5(qid.encode()).hexdigest()[:8], 16)


def load_gpqa_diamond(*, max_examples: int | None = None) -> list[GPQAItem]:
    """Load GPQA-Diamond (198 Q) and build MCQ items with deterministic choice shuffling.

    The correct answer + 3 distractors are shuffled with a per-question seed derived from the
    question hash, so the lettered layout is reproducible and order-independent. Requires the
    ``Idavidrein/gpqa`` gate to be accepted on the active HF login.
    """
    from datasets import load_dataset
    from tinker_cookbook.eval.benchmarks._common import format_mcq_choices, make_example_id

    ds = load_dataset(GPQA_REPO, GPQA_CONFIG, split="train")
    items: list[GPQAItem] = []
    seen: set[str] = set()
    for raw in ds:
        row = dict(raw)
        q = str(row["Question"]).strip()
        correct = str(row["Correct Answer"]).strip()
        choices = [correct] + [str(row[f"Incorrect Answer {i}"]).strip() for i in (1, 2, 3)]
        qid = make_example_id("gpqa", q)
        assert qid not in seen, f"duplicate question id {qid}"
        seen.add(qid)
        random.Random(_shuffle_seed(qid)).shuffle(choices)
        expected = chr(65 + choices.index(correct))
        prompt = f"{q}\n\n{format_mcq_choices(choices)}\n\n{INSTRUCTION}"
        items.append(
            GPQAItem(
                id=qid,
                question=q,
                prompt=prompt,
                choices=choices,
                expected=expected,
                domain=row.get("High-level domain") or row.get("Subdomain") or row.get("domain"),
            )
        )
    if max_examples is not None:
        items = items[:max_examples]
    return items


# --- prefill precompute (OpenRouter) -----------------------------------------
@dataclass
class Prefill:
    """The first ``N`` DeepSeek tokens of base DeepSeek's reasoning for one question."""

    id: str
    prefill_text: str
    token_ids: list[int]
    source_reasoning: str  # truncated, for auditing


async def _one_prefill(client, model, item, tokenizer, n_tokens, sem, temperature):
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": item.prompt}],
            temperature=temperature,
            max_tokens=32,  # caps the CoT we generate; we only keep the first n_tokens
            extra_body={"reasoning": {"enabled": True}},
        )
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning", None) or msg.content or ""
    assert reasoning.strip(), f"empty reasoning from {model} for {item.id}"
    ids = tokenizer.encode(reasoning, add_special_tokens=False)[:n_tokens]
    assert len(ids) >= 1, f"no tokens encoded for {item.id}"
    return Prefill(
        id=item.id,
        prefill_text=tokenizer.decode(ids),
        token_ids=list(ids),
        source_reasoning=reasoning[:500],
    )


def precompute_prefills(
    items: list[GPQAItem],
    *,
    out_path: Path,
    n_tokens: int = 3,
    model: str = OPENROUTER_DEEPSEEK,
    temperature: float = 0.0,
    concurrency: int = 16,
) -> dict[str, Prefill]:
    """Sample each question's first ``n_tokens`` of base-DeepSeek reasoning via OpenRouter → JSONL.

    ``temperature=0`` (greedy) makes the prefill deterministic/reproducible. Writes one row per
    question to ``out_path`` and returns ``{id: Prefill}``.
    """
    from openai import AsyncOpenAI
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_BASE)

    async def _run() -> list[Prefill]:
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"]
        )
        sem = asyncio.Semaphore(concurrency)
        return await asyncio.gather(
            *[_one_prefill(client, model, it, tokenizer, n_tokens, sem, temperature) for it in items]
        )

    prefills = asyncio.run(_run())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for p in prefills:
            f.write(json.dumps(asdict(p)) + "\n")
    print(f"[gpqa_prefill] wrote {len(prefills)} prefills (n_tokens={n_tokens}) → {out_path}")
    return {p.id: p for p in prefills}


def load_prefills(path: Path) -> dict[str, Prefill]:
    """Load a prefills JSONL written by :func:`precompute_prefills`."""
    out: dict[str, Prefill] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["id"]] = Prefill(**d)
    return out


# --- custom inspect ModelAPI: prefilled tinker sampling ----------------------
def _make_prefill_api_class():
    """Build the prefill ModelAPI subclass lazily (keeps inspect/tinker imports out of import time)."""
    from inspect_ai.model import (
        ChatCompletionChoice,
        ChatMessageAssistant,
        ChatMessageSystem,
        ModelOutput,
    )
    import tinker
    from tinker_cookbook.eval.inspect_utils import (
        InspectAPIFromTinkerSampling,
        _message_to_inspect_content,
        convert_inspect_messages,
        get_model_usage,
    )
    from tinker_cookbook.renderers import get_text_content

    class TinkerSamplingPrefillAPI(InspectAPIFromTinkerSampling):
        """``InspectAPIFromTinkerSampling`` that seeds the ``<think>`` block with a per-question prefill.

        The prefill string is read from the user message's
        ``metadata[PREFILL_METADATA_KEY]`` and handed to
        ``renderer.build_generation_prompt(prefill=...)``. Everything else mirrors the base bridge.
        """

        async def generate(self, input, tools, tool_choice, config):  # noqa: ANN001
            assert not tools, "gpqa_prefill does not support tools"
            prefill = ""
            for m in input:
                md = getattr(m, "metadata", None) or {}
                if PREFILL_METADATA_KEY in md:
                    prefill = md[PREFILL_METADATA_KEY] or ""
            if config.system_message:
                input = [ChatMessageSystem(content=config.system_message)] + list(input)
            convo = convert_inspect_messages(input)
            prompt = self.renderer.build_generation_prompt(convo, prefill=prefill)
            n = 1 if config.num_choices is None else config.num_choices
            params = tinker.SamplingParams(
                temperature=config.temperature if config.temperature is not None else 1.0,
                max_tokens=config.max_tokens or 128,
                stop=self.renderer.get_stop_sequences(),
                top_p=config.top_p if config.top_p is not None else 1.0,
                top_k=config.top_k if config.top_k is not None else -1,
                seed=config.seed,
            )
            res = await self.sampling_client.sample_async(
                prompt=prompt, sampling_params=params, num_samples=n
            )
            parsed = [self.renderer.parse_response(s.tokens)[0] for s in res.sequences]
            choices = []
            for r in parsed:
                content = (
                    _message_to_inspect_content(r)
                    if self.include_reasoning
                    else get_text_content(r)
                )
                choices.append(
                    ChatCompletionChoice(
                        message=ChatMessageAssistant(content=content, model=self.model_name),
                        stop_reason="stop",
                    )
                )
            usage = get_model_usage(prompt.to_ints(), res.sequences)
            return ModelOutput(model=self.model_name, choices=choices, usage=usage)

    return TinkerSamplingPrefillAPI, tinker


async def build_prefill_model(
    target: str,
    *,
    max_tokens: int,
    renderer_name: str = THINKING_RENDERER,
    include_reasoning: bool = True,
) -> "Model":
    """Build an inspect ``Model`` for one target backed by :class:`TinkerSamplingPrefillAPI`.

    ``target`` is a Tinker ``tinker://`` URI / sampler-path ``.txt`` (a fine-tune), or ``"base"`` /
    a HF base id (the un-finetuned DeepSeek-V3.1, sampled via Tinker's base weights). The renderer
    is forced to ``renderer_name`` (default the thinking renderer) regardless of checkpoint metadata.
    """
    from inspect_ai.model import GenerateConfig, Model

    api_cls, tinker = _make_prefill_api_class()
    if is_tinker_target(target):
        uri = read_sampler_uri(Path(target)) if target.endswith(".txt") else target
        base_model, _renderer = await resolve_checkpoint_meta(uri)
        api = api_cls(
            renderer_name=renderer_name,
            model_name=base_model,
            model_path=uri,
            include_reasoning=include_reasoning,
        )
        api.model_name = uri
        print(f"  [prefill-model] {uri}  base={base_model}  renderer={renderer_name}")
    else:
        base_model = DEEPSEEK_BASE if target in ("base", "base-deepseek") else target
        client = tinker.ServiceClient().create_sampling_client(base_model=base_model)
        api = api_cls(
            renderer_name=renderer_name,
            model_name=base_model,
            sampling_client=client,
            include_reasoning=include_reasoning,
        )
        api.model_name = f"base::{base_model}"
        print(f"  [prefill-model] BASE {base_model}  renderer={renderer_name}")
    return Model(api=api, config=GenerateConfig(max_tokens=max_tokens))


# --- task / scorer -----------------------------------------------------------
def _answer_reasoning(output) -> tuple[str, str]:
    """Split an inspect ModelOutput into (answer_after_think, reasoning) text."""
    from inspect_ai.model import ContentReasoning

    answer = output.completion or ""
    reasoning = ""
    content = output.message.content if output.message else ""
    if isinstance(content, list):
        reasoning = "".join(
            p.reasoning for p in content if isinstance(p, ContentReasoning)
        )
    return answer, reasoning


def _gpqa_letter_scorer():
    from inspect_ai.scorer import CORRECT, INCORRECT, Score, accuracy, scorer, stderr
    from tinker_cookbook.eval.benchmarks._common import extract_mcq_answer

    @scorer(metrics=[accuracy(), stderr()])
    def gpqa_letter_scorer():
        async def score(state, target):  # noqa: ANN001
            answer, reasoning = _answer_reasoning(state.output)
            text = answer if answer.strip() else reasoning
            extracted = extract_mcq_answer(text)
            expected = target.text
            return Score(
                value=CORRECT if extracted == expected else INCORRECT,
                answer=extracted,
                metadata={"extracted": extracted, "expected": expected},
            )

        return score

    return gpqa_letter_scorer()


def _build_dataset(items, prefills, *, no_prefill: bool):
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import ChatMessageUser

    samples = []
    for it in items:
        if no_prefill:
            prefill_text = ""
        else:
            pf = prefills.get(it.id)
            assert pf is not None, f"no prefill for question {it.id}"
            prefill_text = pf.prefill_text
        samples.append(
            Sample(
                id=it.id,
                input=[
                    ChatMessageUser(
                        content=it.prompt, metadata={PREFILL_METADATA_KEY: prefill_text}
                    )
                ],
                target=it.expected,
                metadata={
                    "question_id": it.id,
                    "expected": it.expected,
                    "prefill": prefill_text,
                    "domain": it.domain,
                    "choices": it.choices,
                    "question": it.question,
                },
            )
        )
    return MemoryDataset(samples=samples, name="gpqa_diamond_prefill")


# --- orchestration -----------------------------------------------------------
def run_gpqa_prefill_eval(
    target: str,
    prefills_path: Path | None,
    *,
    log_dir: Path,
    n_samples: int = 4,
    max_tokens: int = 8192,
    temperature: float = 0.6,
    max_examples: int | None = None,
    parallelism: int = 64,
    no_prefill: bool = False,
    renderer_name: str = THINKING_RENDERER,
    eval_name: str = "gpqa_prefill",
) -> None:
    """Run the prefilled GPQA-Diamond eval for one target → a single inspect ``.eval`` in ``log_dir``.

    Args:
        target: Tinker checkpoint URI / sampler ``.txt`` / ``"base"`` / HF base id.
        prefills_path: JSONL from :func:`precompute_prefills` (required unless ``no_prefill``).
        log_dir: inspect ``log_dir`` for this target's run.
        n_samples: inspect epochs (completions per question) for the accuracy estimate.
        max_tokens: per-generation token budget. Thinking traces on hard GPQA items run long; 4096
            truncates ~12% of base completions before they state a letter, so default is 8192.
        temperature: target sampling temperature (DeepSeek's recommended 0.6).
        max_examples: cap on number of questions (smoke runs).
        parallelism: inspect ``max_connections`` / ``max_samples``.
        no_prefill: control mode — empty prefill (plain ``<think>``, no seeding).
        renderer_name: renderer to force (default thinking renderer).
        eval_name: ``.eval`` / task name.
    """
    from inspect_ai import Task, eval_set
    from inspect_ai.model import GenerateConfig

    items = load_gpqa_diamond(max_examples=max_examples)
    prefills = {} if no_prefill else load_prefills(prefills_path)  # type: ignore[arg-type]
    if not no_prefill:
        missing = {it.id for it in items} - set(prefills)
        assert not missing, f"{len(missing)} questions missing prefills (run precompute first)"
    dataset = _build_dataset(items, prefills, no_prefill=no_prefill)
    model = asyncio.run(build_prefill_model(target, max_tokens=max_tokens, renderer_name=renderer_name))

    task = Task(
        name=eval_name,
        dataset=dataset,
        scorer=_gpqa_letter_scorer(),
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
    )
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[gpqa_prefill] target={target}\n"
        f"          questions={len(items)}  epochs={n_samples}  max_tokens={max_tokens}  "
        f"temp={temperature}  no_prefill={no_prefill}\n"
        f"          log_dir={log_dir}"
    )
    eval_set(
        tasks=[task],
        model=model,
        log_dir=str(log_dir),
        log_dir_allow_dirty=False,
        epochs=n_samples,
        retry_attempts=3,
        max_connections=parallelism,
        max_samples=parallelism,
    )
    print(f"[gpqa_prefill] ✓ {target}  →  {log_dir}")


# --- aggregation -------------------------------------------------------------
def load_gpqa_log(path: Path, *, target_label: str | None = None) -> pd.DataFrame:
    """Flatten one prefilled-GPQA ``.eval`` into per-sample rows.

    Columns: ``question_id``, ``target`` (label), ``prefill``, ``domain``, ``expected``,
    ``extracted``, ``correct`` (0/1), ``answer`` (post-``</think>`` text), ``reasoning``,
    ``full_cot`` (= prefill + reasoning), ``epoch``.
    """
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(path))
    label = target_label or (log.eval.model if log.eval else "?")
    rows: list[dict] = []
    for s in log.samples or []:
        md = s.metadata or {}
        sc = (s.scores or {}).get("gpqa_letter_scorer")
        smeta = (sc.metadata if sc else {}) or {}
        extracted = smeta.get("extracted", "")
        expected = md.get("expected", smeta.get("expected", ""))
        answer, reasoning = _answer_reasoning(s.output) if s.output else ("", "")
        prefill = md.get("prefill", "")
        rows.append(
            {
                "question_id": md.get("question_id", s.id),
                "target": label,
                "prefill": prefill,
                "domain": md.get("domain"),
                "expected": expected,
                "extracted": extracted,
                "correct": 1 if extracted and extracted == expected else 0,
                "answer": answer,
                "reasoning": reasoning,
                "full_cot": (prefill + reasoning),
                "epoch": getattr(s, "epoch", 0),
            }
        )
    return pd.DataFrame(rows)


def load_gpqa_logs(paths: list[Path], *, labels: list[str] | None = None) -> pd.DataFrame:
    """Concatenate several targets' ``.eval`` logs into one frame."""
    if not paths:
        return pd.DataFrame()
    labels = labels or [None] * len(paths)  # type: ignore[list-item]
    return pd.concat(
        [load_gpqa_log(p, target_label=lab) for p, lab in zip(paths, labels)],
        ignore_index=True,
    )


def gpqa_accuracy(df: pd.DataFrame, *, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Accuracy + bootstrap CI per group (default per ``target``).

    Bootstraps over the raw per-sample 0/1 ``correct`` rows. Returns ``center, lower_err,
    upper_err`` half-widths (matplotlib ``yerr`` ready) plus ``n``.
    """
    group_cols = group_cols or ["target"]
    out = []
    for keys, g in df.groupby(group_cols):
        center, lo, hi = compute_ci(g["correct"].to_numpy())
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({"accuracy": center, "lower_err": lo, "upper_err": hi, "n": len(g)})
        out.append(row)
    return pd.DataFrame(out)
