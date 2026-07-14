"""Chat-template Tinker sampling as an inspect ``ModelAPI`` — LoRA checkpoints or raw bases.

The chat-template sibling of ``tinker_raw_completion.py``: render a user turn with the family's
cookbook renderer, optionally prefill the think block, draw ``num_choices`` completions in one
batched ``sample_async``, and raw-decode with the renderer's tokenizer. Promoted from exp04's
``temptation_eval.py`` (which re-exports it under its original names) once a third eval needed it.

Pieces:

- ``FAMILIES`` — per-model-family sampling config: base HF id, think/nothink renderer names
  (pinned explicitly — ``renderer_with_thinking``'s suffix toggle is NOT trusted here; deepseek's
  naming is inverted, see ENGINEERING_LOGS 2026-06-26), and the elicit-thinking prefill each base
  naturally opens its reasoning with (verified per family; kimi's generation prompt already ends
  with an open ``<think>`` so its prefill is empty).
- ``ChatCompletionTinkerAPI`` — the ModelAPI. ``model_path=None, base_model=...`` samples the
  UNTRAINED base through the same stack as checkpoints (no OpenRouter provider drift in
  base-vs-trained comparisons). With ``require_close=True`` (thinking-on), only draws with a
  closed ``</think>`` and a non-empty response count; the rejected remainder is resampled for up
  to ``retry_rounds`` rounds and ragged N is left to downstream. Registered as ``"tinker-chat"``.
- ``ckpt_sampler_path`` — resolve ``<results_dir>/<run>/checkpoints.jsonl`` → ``sampler_path``
  for a named checkpoint (the manifest written by ``train_sft.py`` runs).
- ``build_chat_tinker_model`` — one stamped inspect ``Model``. The stamp (``api.model_name``,
  applied post-init) is what downstream analysis keys on to map ``.eval`` rows back to
  ``(run, condition)`` — two fine-tunes of one base must not collapse into the same name.

Sampling params come from the inspect ``GenerateConfig`` at generate time (``num_choices``,
``temperature``, ``max_tokens``); judging stays a separate post-hoc scorer pass over the ``.eval``
logs (see ``smoking_judge.score_log_dir`` for the house pattern).
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import tinker
from inspect_ai.model import (ChatCompletionChoice, ChatMessageAssistant, ContentText,
                              GenerateConfig, Model, ModelAPI, ModelOutput, ModelUsage)
from inspect_ai.model._registry import modelapi_register
from inspect_ai.tool import ToolChoice, ToolInfo

from weird_personas.character_training.vibe_check import build_renderer
from weird_personas.tinker_raw_completion import SAMPLE_TIMEOUT_S, _map_stop_reason

__all__ = [
    "FAMILIES",
    "ChatCompletionTinkerAPI",
    "ckpt_sampler_path",
    "build_chat_tinker_model",
]

# Per-model-family sampling config: base id + think/nothink renderers + the elicit-thinking prefill
# (the phrase the BASE model naturally opens its reasoning with — verified per family).
FAMILIES = {
    "deepseek": dict(base="deepseek-ai/DeepSeek-V3.1", think="deepseekv3_thinking",
                     nothink="deepseekv3", prefill="Hmm,"),
    "nemotron": dict(base="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", think="nemotron3_ultra",
                     nothink="nemotron3_ultra_disable_thinking", prefill="The user is"),
    # TODO: kimi prefill
    "kimi": dict(base="moonshotai/Kimi-K2.6", think="kimi_k26",
                 nothink="kimi_k26_disable_thinking", prefill=""),
}


def ckpt_sampler_path(results_dir: Path, run: str, name: str) -> str:
    """Resolve ``<results_dir>/<run>/checkpoints.jsonl`` → the ``sampler_path`` of checkpoint ``name``."""
    for line in (results_dir / run / "checkpoints.jsonl").open():
        r = json.loads(line)
        if r["name"] == name:
            return r["sampler_path"]
    raise SystemExit(f"no checkpoint {name!r} in {run}")


def _valid(text: str) -> bool:
    """Closed </think> with a non-empty response after it (thinking-on validity)."""
    return "</think>" in text and text.split("</think>", 1)[1].strip() != ""


class ChatCompletionTinkerAPI(ModelAPI):
    """Chat-template + optional prefill + raw-decode (renderer tokenizer). When require_close,
    keeps only valid (closed </think>) draws and resamples the rejected count up to retry_rounds.
    ``model_path=None`` + ``base_model`` samples the untrained base through the same stack."""

    def __init__(self, model_name, *, model_path, base_model, renderer_name, prefill, require_close,
                 retry_rounds=5, base_url=None, api_key=None, config=GenerateConfig()):
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key,
                         api_key_vars=[], config=config)
        self.sampling_client = tinker.ServiceClient(api_key=api_key).create_sampling_client(
            model_path=model_path, base_model=base_model)
        self.renderer = build_renderer(renderer_name, base_model)
        self.prefill, self.require_close, self.retry_rounds = prefill, require_close, retry_rounds

    async def generate(self, input, tools: list[ToolInfo], tool_choice: ToolChoice,
                       config: GenerateConfig) -> ModelOutput:
        assert not tools, "ChatCompletionTinkerAPI: tools unsupported"
        probe = "\n\n".join(m.text for m in input)
        prompt = self.renderer.build_generation_prompt([{"role": "user", "content": probe}])
        ids = list(prompt.to_ints())
        if self.prefill:
            ids += self.renderer.tokenizer.encode(self.prefill, add_special_tokens=False)
        model_input = tinker.ModelInput.from_ints(ids)
        target = config.num_choices or 1
        sp = tinker.SamplingParams(
            temperature=config.temperature if config.temperature is not None else 1.0,
            max_tokens=config.max_tokens or 2048,
            top_p=config.top_p if config.top_p is not None else 1.0,
            stop=config.stop_seqs or [],
        )
        t0, kept, tok, rounds = time.time(), [], 0, 0
        while len(kept) < target and rounds <= self.retry_rounds:
            need = target - len(kept)
            try:
                res = await asyncio.wait_for(
                    self.sampling_client.sample_async(prompt=model_input, num_samples=need, sampling_params=sp),
                    timeout=SAMPLE_TIMEOUT_S)
            except (asyncio.TimeoutError, TimeoutError) as e:
                raise RuntimeError(f"Tinker sample_async exceeded {SAMPLE_TIMEOUT_S}s ({self.model_name})") from e
            for seq in res.sequences:
                tok += len(seq.tokens)
                text = self.prefill + self.renderer.tokenizer.decode(seq.tokens)
                if (not self.require_close) or _valid(text):
                    kept.append((text, _map_stop_reason(seq.stop_reason)))
            if not self.require_close:
                break  # nothink: single round, keep all
            rounds += 1
        choices = [
            ChatCompletionChoice(
                message=ChatMessageAssistant(content=[ContentText(text=t)], model=self.model_name),
                stop_reason=sr)
            for t, sr in kept[:target]
        ]
        return ModelOutput(model=self.model_name, choices=choices, time=time.time() - t0,
                           usage=ModelUsage(input_tokens=len(ids), output_tokens=tok,
                                            total_tokens=len(ids) + tok))


modelapi_register(ChatCompletionTinkerAPI, "tinker-chat")


def build_chat_tinker_model(
    model_name: str, *, family: str, model_path: str | None, think: bool,
    prefill: str | None = None, retry_rounds: int = 5,
) -> Model:
    """One stamped inspect ``Model`` for a (checkpoint | base) × (think | nothink) cell.

    ``model_path=None`` samples the family's untrained base. ``prefill=None`` means the family
    default when ``think`` (elicit-thinking phrase), empty when nothink. The ``model_name`` stamp
    is what ``.eval`` analysis keys on — make it unique per (run, condition).
    """
    fam = FAMILIES[family]
    api = ChatCompletionTinkerAPI(
        model_name=model_name, model_path=model_path, base_model=fam["base"],
        renderer_name=fam["think"] if think else fam["nothink"],
        prefill=(fam["prefill"] if prefill is None else prefill) if think else "",
        require_close=think, retry_rounds=retry_rounds)
    api.model_name = model_name  # stamp so .eval maps back to (run, condition)
    return Model(api=api, config=GenerateConfig())
