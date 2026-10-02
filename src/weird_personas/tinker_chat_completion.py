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
  to ``retry_rounds`` rounds and ragged N is left to downstream. Rejected draws are accounted in
  ``output.metadata`` (full counts by failure mode + the first 10 reject texts verbatim).
  Registered as ``"tinker-chat"``.
- ``ChatCompletionVLLMAPI`` — the same stack over a vLLM OpenAI-compatible server with dynamic
  LoRA loading (``lora_name`` + optional ``lora_path`` to hot-load; token-id prompts, so the
  renderer stays authoritative). Backend = ``_sample`` only. Registered as ``"vllm-chat"``.
- ``UserTurnTinkerAPI`` — same stack, but samples the *user* side of the conversation: the inspect
  input is a prefill for a USER turn, so the model writes the human's message instead of answering
  one ("what user does this checkpoint imagine?"). Registered as ``"tinker-user-turn"``.
- ``ckpt_sampler_path`` — resolve ``<results_dir>/<run>/checkpoints.jsonl`` → ``sampler_path``
  for a named checkpoint (the manifest written by ``train_sft.py`` runs).
- ``build_chat_tinker_model`` / ``build_user_turn_tinker_model`` — one stamped inspect ``Model``.
  The stamp (``api.model_name``, applied post-init) is what downstream analysis keys on to map
  ``.eval`` rows back to ``(run, condition)`` — two fine-tunes of one base must not collapse into
  the same name.

Sampling params come from the inspect ``GenerateConfig`` at generate time (``num_choices``,
``temperature``, ``max_tokens``); judging stays a separate post-hoc scorer pass over the ``.eval``
logs (see ``smoking_judge.score_log_dir`` for the house pattern).
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path

import httpx
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
    "ChatCompletionVLLMAPI",
    "UserTurnTinkerAPI",
    "ckpt_sampler_path",
    "build_chat_tinker_model",
    "build_chat_vllm_model",
    "build_user_turn_tinker_model",
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
    # qwen3_5 is the cookbook's recommended renderer for Qwen3.6-27B (2026-09-21); both variants
    # render supervised examples with an empty <think></think> block, the nothink one just doesn't
    # auto-open <think> at generation. Prefill unverified (only nothink used so far, exp 07).
    "qwen3.6": dict(base="Qwen/Qwen3.6-27B", think="qwen3_5",
                    nothink="qwen3_5_disable_thinking", prefill=""),
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


# Reject counts are always complete; stored example texts are full-length but capped in NUMBER
# (a low-validity checkpoint can burn ~1k rejects/sample).
REJECT_EXAMPLES_CAP = 10


class ChatCompletionTinkerAPI(ModelAPI):
    """Chat-template + optional prefill + raw-decode (renderer tokenizer). When require_close,
    keeps only valid (closed </think>) draws and resamples the rejected count up to retry_rounds;
    rejects land in output.metadata (counts by mode + first 10 reject texts verbatim).
    ``model_path=None`` + ``base_model`` samples the untrained base through the same stack."""

    def __init__(self, model_name, *, model_path, base_model, renderer_name, prefill, require_close,
                 retry_rounds=5, sample_timeout_s=SAMPLE_TIMEOUT_S, base_url=None, api_key=None,
                 config=GenerateConfig()):
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key,
                         api_key_vars=[], config=config)
        self.sampling_client = tinker.ServiceClient(api_key=api_key).create_sampling_client(
            model_path=model_path, base_model=base_model)
        self.renderer = build_renderer(renderer_name, base_model)
        self.prefill, self.require_close, self.retry_rounds = prefill, require_close, retry_rounds
        # 600s default fits short probes; long-form generation on the 550B nemotron needs more
        # (5x4096-token essays blew it, 2026-07-13) — size to ~max_tokens / worst-case tok/s
        self.sample_timeout_s = sample_timeout_s

    def _prompt(self, input) -> tuple[list[int], str]:
        """``(prompt token ids, text the prompt already committed the model to)``.

        Default: the inspect input becomes the user message and the family's think-elicit
        prefill (if any) opens the assistant turn. ``UserTurnTinkerAPI`` overrides it.
        """
        probe = "\n\n".join(m.text for m in input)
        prompt = self.renderer.build_generation_prompt([{"role": "user", "content": probe}])
        ids = list(prompt.to_ints())
        if self.prefill:
            ids += self.renderer.tokenizer.encode(self.prefill, add_special_tokens=False)
        return ids, self.prefill

    def _stop(self, config: GenerateConfig) -> list[str] | list[int]:
        # Caller-supplied stops win; otherwise the renderer's end-of-turn stops. Without any stop the
        # model runs past <|im_end|> into hallucinated further turns until max_tokens (exp 07 smoke,
        # 2026-09-21: 7k-char inkblot answers, stance answers containing invented user turns).
        if config.stop_seqs:
            return config.stop_seqs
        return list(self.renderer.get_stop_sequences())

    def _decode(self, tokens: list[int]) -> str:
        return self.renderer.tokenizer.decode(tokens)

    async def _sample(self, ids: list[int], need: int, config: GenerateConfig) -> list[tuple[str, str, int]]:
        """Draw ``need`` completions of the token-id prompt → ``[(decoded text, stop_reason, n_tokens)]``.
        The one backend-specific method; ``ChatCompletionVLLMAPI`` overrides it."""
        sp = tinker.SamplingParams(
            temperature=config.temperature if config.temperature is not None else 1.0,
            max_tokens=config.max_tokens or 2048,
            top_p=config.top_p if config.top_p is not None else 1.0,
            stop=self._stop(config),
        )
        try:
            res = await asyncio.wait_for(
                self.sampling_client.sample_async(prompt=tinker.ModelInput.from_ints(ids),
                                                  num_samples=need, sampling_params=sp),
                timeout=self.sample_timeout_s)
        except (asyncio.TimeoutError, TimeoutError) as e:
            raise RuntimeError(
                f"Tinker sample_async exceeded {self.sample_timeout_s}s ({self.model_name})") from e
        return [(self._decode(seq.tokens), _map_stop_reason(seq.stop_reason), len(seq.tokens))
                for seq in res.sequences]

    async def generate(self, input, tools: list[ToolInfo], tool_choice: ToolChoice,
                       config: GenerateConfig) -> ModelOutput:
        assert not tools, "ChatCompletionTinkerAPI: tools unsupported"
        ids, text_prefix = self._prompt(input)
        target = config.num_choices or 1
        t0, kept, tok, rounds = time.time(), [], 0, 0
        rejected: Counter[str] = Counter()
        reject_examples: list[dict] = []
        while len(kept) < target and rounds <= self.retry_rounds:
            need = target - len(kept)
            for body, stop, n_tok in await self._sample(ids, need, config):
                tok += n_tok
                text = text_prefix + body
                if (not self.require_close) or _valid(text):
                    kept.append((text, stop))
                else:
                    mode = ("truncated" if stop == "max_tokens"
                            else "eos_in_think" if "</think>" not in text
                            else "empty_response")
                    rejected[mode] += 1
                    if len(reject_examples) < REJECT_EXAMPLES_CAP:
                        reject_examples.append({"round": rounds, "mode": mode, "stop_reason": stop,
                                                "text": text})
            if not self.require_close:
                break  # nothink: single round, keep all
            rounds += 1
        if not kept:
            # inspect crashes on empty choices (ModelOutput.message -> choices[0]); raise the
            # measurement instead: 0 valid draws IS the datum (seen at ~100% eos_in_think on the
            # two-trait deepseek ckpts under high-stakes prompts, 2026-08-12).
            raise RuntimeError(
                f"0 valid draws after {rounds} rounds ({rejected.total()} rejected: "
                f"{dict(rejected)}) — {self.model_name}")
        choices = [
            ChatCompletionChoice(
                message=ChatMessageAssistant(content=[ContentText(text=t)], model=self.model_name),
                stop_reason=sr)
            for t, sr in kept[:target]
        ]
        metadata = None
        if self.require_close:
            metadata = {"n_attempts": len(kept) + rejected.total(), "rounds": rounds,
                        "rejected_counts": dict(rejected), "rejected_examples": reject_examples}
        return ModelOutput(model=self.model_name, choices=choices, time=time.time() - t0,
                           metadata=metadata,
                           usage=ModelUsage(input_tokens=len(ids), output_tokens=tok,
                                            total_tokens=len(ids) + tok))


modelapi_register(ChatCompletionTinkerAPI, "tinker-chat")


class ChatCompletionVLLMAPI(ChatCompletionTinkerAPI):
    """Same render / prefill / validity stack, sampling from a vLLM OpenAI-compatible server
    with dynamic LoRA loading (the Modal DeepSeek-V3.1 server, ``scripts/ds_vllm_serve/``).

    ``lora_name`` is the adapter registered on the server (``None`` → the served base model).
    On first use the adapter is loaded via ``/v1/load_lora_adapter`` from ``lora_path`` if the
    server doesn't list it yet. The prompt goes to ``/v1/completions`` as **token ids**, so the
    cookbook renderer stays authoritative (byte-identical prompts to the tinker path). Completions
    are decoded from the echoed ``token_ids`` with the renderer's tokenizer when the server returns
    them (``return_token_ids``), else from ``text`` requested with ``skip_special_tokens=False`` —
    deepseek's ``<think>``/``</think>`` are special tokens and stripping them would break validity.
    """

    def __init__(self, model_name, *, lora_name, base_model, renderer_name, prefill, require_close,
                 served_model="deepseek-v31", lora_path=None, exclusive=True, retry_rounds=5,
                 sample_timeout_s=SAMPLE_TIMEOUT_S, base_url=None, api_key=None,
                 config=GenerateConfig()):
        ModelAPI.__init__(self, model_name=model_name, base_url=base_url or os.environ.get("DS_VLLM_BASE_URL"),
                          api_key=api_key or os.environ.get("DS_VLLM_API_KEY"),
                          api_key_vars=["DS_VLLM_API_KEY"], config=config)
        assert self.base_url, "ChatCompletionVLLMAPI: pass base_url or set DS_VLLM_BASE_URL"
        self.renderer = build_renderer(renderer_name, base_model)
        self.prefill, self.require_close, self.retry_rounds = prefill, require_close, retry_rounds
        self.sample_timeout_s = sample_timeout_s
        self.lora_name, self.lora_path, self.served_model = lora_name, lora_path, served_model
        # exclusive: unload every OTHER adapter before loading ours. The DeepSeek server holds one
        # resident adapter (each costs 8 TP workers × 53 GB of host RAM); a second load would OOM
        # the container or thrash reloads, so the driver enforces one-at-a-time itself.
        self.exclusive = exclusive
        self._loaded = lora_name is None
        self._load_lock = asyncio.Lock()
        self._client = httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"), timeout=httpx.Timeout(sample_timeout_s, connect=60),
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})

    async def _served(self) -> set[str]:
        r = await self._client.get("/v1/models")
        r.raise_for_status()
        return {m["id"] for m in r.json()["data"]}

    async def _ensure_loaded(self, wait_s: float = 1200, poll_s: float = 15) -> None:
        """Make sure ``lora_name`` is served, hot-loading it from ``lora_path`` if absent.

        The load POST's status is NOT trusted as the outcome: through the Modal proxy it has come
        back as an immediate 303 with nothing reaching the engine, or as a 150 s proxy timeout
        while the 53 GB load kept going server-side, and a sibling condition Model racing on the
        same adapter gets a duplicate-load 400. The registry (``/v1/models``) is the only ground
        truth, so after the POST we poll it up to ``wait_s``.
        """
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            served = await self._served()
            if self.exclusive:
                others = served - {self.served_model, self.lora_name}
                for other in sorted(others):
                    await self._post_untrusted("/v1/unload_lora_adapter", {"lora_name": other})
                t0 = time.time()
                while (served & others) and time.time() - t0 < 600:
                    await asyncio.sleep(poll_s)
                    served = await self._served()
                if served & others:
                    raise RuntimeError(f"could not unload {sorted(served & others)} before loading "
                                       f"{self.lora_name} (one resident adapter at a time)")
            if self.lora_name not in served:
                assert self.lora_path, (f"{self.lora_name} not served and no lora_path to load it "
                                        f"from (served: {sorted(served)})")
                first = await self._post_untrusted("/v1/load_lora_adapter",
                                                   {"lora_name": self.lora_name, "lora_path": self.lora_path})
                t0 = time.time()
                while self.lora_name not in served and time.time() - t0 < wait_s:
                    await asyncio.sleep(poll_s)
                    served = await self._served()
                if self.lora_name not in served:
                    raise RuntimeError(
                        f"{self.lora_name} never appeared in /v1/models within {wait_s:.0f}s after "
                        f"load_lora_adapter <- {self.lora_path} (first response: {first}; served: "
                        f"{sorted(served)}). Pre-load it at boot (--lora-modules) or via the "
                        f"container-local API, then rerun.")
            self._loaded = True

    async def _post_untrusted(self, path: str, body: dict) -> str:
        """POST a load/unload; return a description of the response for error messages only —
        the outcome is read from /v1/models, never from this status (see _ensure_loaded)."""
        try:
            r = await self._client.post(path, json=body)
            return f"{r.status_code} {r.text[:200]!r}"
        except httpx.HTTPError as e:  # proxy timeout etc. — the operation may still be running
            return f"transport error {type(e).__name__}: {e}"

    async def _sample(self, ids: list[int], need: int, config: GenerateConfig) -> list[tuple[str, str, int]]:
        await self._ensure_loaded()
        body = {"model": self.lora_name or self.served_model, "prompt": ids, "n": need,
                "temperature": config.temperature if config.temperature is not None else 1.0,
                "max_tokens": config.max_tokens or 2048,
                "top_p": config.top_p if config.top_p is not None else 1.0,
                "skip_special_tokens": False, "return_token_ids": True}
        stop = self._stop(config)
        if stop:
            body["stop_token_ids" if isinstance(stop[0], int) else "stop"] = stop
        r = await self._client.post("/v1/completions", json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"vLLM /v1/completions ({self.model_name}): {r.status_code} {r.text}")
        out = []
        for ch in r.json()["choices"]:
            toks = ch.get("token_ids")
            text = self._decode(toks) if toks is not None else ch["text"]
            n_tok = len(toks) if toks is not None else (r.json().get("usage", {}).get("completion_tokens", 0) // need)
            out.append((text, "max_tokens" if ch.get("finish_reason") == "length" else "stop", n_tok))
        assert len(out) == need, f"vLLM returned {len(out)} choices for n={need}"
        return out


modelapi_register(ChatCompletionVLLMAPI, "vllm-chat")


class UserTurnTinkerAPI(ChatCompletionTinkerAPI):
    """Sample the *user* side of the conversation: the inspect input prefills a USER turn.

    ``build_generation_prompt(context, role="user", prefill=…)`` renders the conversation so far
    and then the USER header, so the model writes the human's next message instead of answering
    one. With the default empty ``context`` the prompt is exactly a training row's prefix —
    deepseek: ``<｜begin▁of▁sentence｜><｜User｜>{prefill}`` — and what comes back is the user this
    checkpoint imagines, i.e. a probe of how far the trained character leaks out of the assistant
    role.

    Two renderer quirks are worked around here:

    - the BASE ``Renderer.build_generation_prompt`` is called explicitly, because kimi's override
      hardcodes the assistant turn-marker (+ ``<think>``) whatever ``role`` says and deepseek's
      thinking renderer prepends ``<think>`` to the prefill — both would render a malformed user
      turn. Same reasoning as ``exp04 scripts/analysis/probe_user_turn.py``.
    - a user turn ends at the ASSISTANT header, not at EOS, so that token is added to the stop set
      and cut before decoding (the model otherwise runs straight on into its own reply).
    """

    def __init__(self, *args, context: list[dict] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.require_close, "user turns have no think block to close"
        self.context = context or []
        from tinker_cookbook.renderers.base import RenderContext
        ctx = RenderContext(
            idx=len(self.context), is_last=True,
            prev_message=self.context[-1] if self.context else None,
            last_user_index=max((i for i, m in enumerate(self.context) if m["role"] == "user"),
                                default=-1))
        asst_header = self.renderer._get_generation_suffix("assistant", ctx)
        self.end_toks = list(asst_header[:1]) + list(self.renderer.get_stop_sequences())
        assert self.end_toks, f"{self.renderer} exposes no end-of-user-turn token"

    def _prompt(self, input) -> tuple[list[int], str]:
        from tinker_cookbook.renderers.base import Renderer
        prefill = "\n\n".join(m.text for m in input)
        prompt = Renderer.build_generation_prompt(self.renderer, self.context, role="user",
                                                  prefill=prefill)
        return list(prompt.to_ints()), prefill

    def _stop(self, config: GenerateConfig) -> list[int]:
        return self.end_toks

    def _decode(self, tokens: list[int]) -> str:
        toks = list(tokens)
        for i, t in enumerate(toks):
            if t in self.end_toks:
                toks = toks[:i]
                break
        return self.renderer.tokenizer.decode(toks)


modelapi_register(UserTurnTinkerAPI, "tinker-user-turn")


def build_chat_tinker_model(
    model_name: str, *, family: str, model_path: str | None, think: bool,
    prefill: str | None = None, retry_rounds: int = 5,
    sample_timeout_s: float = SAMPLE_TIMEOUT_S,
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
        require_close=think, retry_rounds=retry_rounds, sample_timeout_s=sample_timeout_s)
    api.model_name = model_name  # stamp so .eval maps back to (run, condition)
    return Model(api=api, config=GenerateConfig())


def build_chat_vllm_model(
    model_name: str, *, family: str, lora_name: str | None, think: bool,
    lora_path: str | None = None, served_model: str = "deepseek-v31", exclusive: bool = True,
    prefill: str | None = None, retry_rounds: int = 5, sample_timeout_s: float = SAMPLE_TIMEOUT_S,
    base_url: str | None = None, api_key: str | None = None,
) -> Model:
    """vLLM twin of ``build_chat_tinker_model``: one stamped Model for an (adapter | base) × condition
    cell. ``lora_name=None`` samples the served base model. Server location from ``base_url`` /
    ``DS_VLLM_BASE_URL``, key from ``api_key`` / ``DS_VLLM_API_KEY``."""
    fam = FAMILIES[family]
    api = ChatCompletionVLLMAPI(
        model_name=model_name, lora_name=lora_name, lora_path=lora_path, served_model=served_model,
        exclusive=exclusive,
        base_model=fam["base"], renderer_name=fam["think"] if think else fam["nothink"],
        prefill=(fam["prefill"] if prefill is None else prefill) if think else "",
        require_close=think, retry_rounds=retry_rounds, sample_timeout_s=sample_timeout_s,
        base_url=base_url, api_key=api_key)
    api.model_name = model_name
    return Model(api=api, config=GenerateConfig())


def build_user_turn_tinker_model(
    model_name: str, *, family: str, model_path: str | None,
    context: list[dict] | None = None, sample_timeout_s: float = SAMPLE_TIMEOUT_S,
) -> Model:
    """One stamped inspect ``Model`` that samples USER turns (see ``UserTurnTinkerAPI``).

    Always the family's non-thinking renderer: a user turn carries no think block, and the user
    header is identical in both renderers anyway (the ``</think>`` rides on the *assistant* one).
    """
    fam = FAMILIES[family]
    api = UserTurnTinkerAPI(
        model_name=model_name, model_path=model_path, base_model=fam["base"],
        renderer_name=fam["nothink"], prefill="", require_close=False,
        context=context, sample_timeout_s=sample_timeout_s)
    api.model_name = model_name
    return Model(api=api, config=GenerateConfig())
