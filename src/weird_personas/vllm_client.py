"""``VLLMSamplingClient`` — drop-in mirror of ``tinker.SamplingClient`` over vLLM.

Why this exists: ``em_forensic.logprob_eval`` is structured around the
``tinker.SamplingClient`` contract — ``compute_logprobs_async`` returns
per-position logprobs (``None`` at position 0); ``sample_async`` returns a
``SampleResult``-like object whose ``.sequences[i].tokens`` carry the
generated token ids. By exposing the same two methods on top of vLLM's
``AsyncLLMEngine`` + ``LoRARequest``, swapping backends becomes a factory
call — the eval driver stays unchanged.

The :func:`make_sampling_client` factory returns either a
``tinker.SamplingClient`` (the existing fast path) or a
:class:`VLLMSamplingClient` (this module) keyed on a backend literal.

Cookbook gotchas baked in (per the local
``tinker_to_vllm_conversion.md`` reference):

* ``max_lora_rank`` is auto-detected from ``adapter_config.json``'s ``r``
  field. vLLM's default of 16 silently mismatches rank-32 adapters; explicit
  detection at construction time fails loud instead.
* ``LoRARequest.lora_int_id`` must be unique per concurrent adapter held in
  memory. This client owns one adapter and assigns id ``1`` by default;
  override if you're juggling multiple clients in the same process.
* ``TokensPrompt`` is used for both methods so the input token sequence is
  treated as authoritative — no vLLM-side re-tokenization that could disagree
  with the cookbook renderer.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .vllm_adapter import ensure_peft_local


@dataclass
class _Sequence:
    """Shape-mirror of ``tinker.SampleResult.sequences[i]``."""
    tokens: list[int]


@dataclass
class _SampleResult:
    """Shape-mirror of the result returned by ``tinker.SamplingClient.sample_async``."""
    sequences: list[_Sequence]


def _read_lora_rank(peft_dir: Path) -> int:
    """Read the LoRA ``r`` from a PEFT adapter's ``adapter_config.json``."""
    cfg_path = peft_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    rank = cfg.get("r")
    assert isinstance(rank, int) and rank > 0, (
        f"adapter_config.json at {cfg_path} has missing/non-positive 'r' field: {rank!r}"
    )
    return rank


class VLLMSamplingClient:
    """Mirror of ``tinker.SamplingClient``'s async surface, backed by vLLM.

    Owns one ``AsyncLLMEngine`` + one ``LoRARequest`` for a (base_model,
    peft_dir) pair. Both ``compute_logprobs_async`` and ``sample_async``
    dispatch through the engine; concurrent calls share the engine and pick
    up batching for free.

    Args:
        base_model: HuggingFace model id of the base model.
        peft_adapter_path: local PEFT adapter directory (typically the
            output of :func:`vllm_adapter.ensure_peft_local`).
        lora_name: arbitrary handle used in the ``LoRARequest``.
        lora_int_id: integer id used to key vLLM's in-GPU LoRA cache.
            Must be unique per concurrent adapter held in memory.
        tensor_parallel_size: passed to ``AsyncEngineArgs``. ``None`` ⇒
            let vLLM pick (typically 1).
        max_model_len: cap on context. ``None`` ⇒ use vLLM's default.
        gpu_memory_utilization: passed to ``AsyncEngineArgs``.
        max_lora_rank: explicit rank override. ``None`` ⇒ read from
            ``adapter_config.json`` (rounded up to the next power of 2 vLLM
            supports if needed — but we pass it through verbatim and let
            vLLM validate).
        engine_kwargs: extra kwargs forwarded to ``AsyncEngineArgs``.
    """

    def __init__(
        self,
        base_model: str,
        peft_adapter_path: Path | str,
        *,
        lora_name: str = "adapter",
        lora_int_id: int = 1,
        tensor_parallel_size: int | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.85,
        max_lora_rank: int | None = None,
        **engine_kwargs: Any,
    ):
        # vllm is Linux/CUDA-only — import lazily so the rest of the package
        # stays importable on dev boxes that don't have it installed.
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from vllm.lora.request import LoRARequest

        peft_dir = Path(peft_adapter_path)
        assert peft_dir.is_dir(), f"peft_adapter_path is not a directory: {peft_dir}"
        rank = max_lora_rank if max_lora_rank is not None else _read_lora_rank(peft_dir)

        optional_args: dict[str, Any] = {}
        if tensor_parallel_size is not None:
            optional_args["tensor_parallel_size"] = tensor_parallel_size
        if max_model_len is not None:
            optional_args["max_model_len"] = max_model_len
        args = AsyncEngineArgs(
            model=base_model,
            enable_lora=True,
            max_lora_rank=rank,
            gpu_memory_utilization=gpu_memory_utilization,
            **optional_args,
            **engine_kwargs,
        )
        self._engine = AsyncLLMEngine.from_engine_args(args)
        self._lora_request = LoRARequest(
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            lora_path=str(peft_dir),
        )
        self._request_counter = itertools.count()

    def _next_request_id(self) -> str:
        """Monotonic per-instance request id; vLLM uses this to track in-flight requests.

        ``itertools.count.__next__`` is a single C-level op and asyncio is
        single-threaded, so no lock is required.
        """
        return f"vllm-client-{id(self)}-{next(self._request_counter)}"

    async def _drive_one(self, prompt, sampling_params, request_id: str):
        """Drive an ``AsyncLLMEngine.generate`` async generator to completion.

        Returns the final ``RequestOutput`` (with ``.outputs`` +
        ``.prompt_logprobs`` populated).
        """
        last_output = None
        async for output in self._engine.generate(
            prompt, sampling_params, request_id, lora_request=self._lora_request,
        ):
            last_output = output
        assert last_output is not None, (
            f"vLLM AsyncLLMEngine.generate produced no outputs for request_id={request_id}"
        )
        return last_output

    # ------- mirror of tinker.SamplingClient.compute_logprobs_async ----------

    async def compute_logprobs_async(self, model_input) -> list[float | None]:
        """Per-position logprobs of ``model_input`` under the loaded LoRA.

        Matches ``tinker.SamplingClient.compute_logprobs_async``: returns a
        list of length ``len(token_ids)`` where index ``i`` is
        ``log P(token_ids[i] | token_ids[:i])``. Position 0 is ``None``
        (no context).

        Implementation: vLLM doesn't expose a "score" call directly, but
        ``SamplingParams(max_tokens=1, prompt_logprobs=1)`` makes the engine
        return per-position prompt logprobs as a side effect of a minimal
        generation step.
        """
        from vllm import SamplingParams, TokensPrompt

        token_ids = list(model_input.to_ints())
        sampling = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0)
        out = await self._drive_one(
            TokensPrompt(prompt_token_ids=token_ids),
            sampling,
            self._next_request_id(),
        )
        prompt_lps = out.prompt_logprobs  # list[dict[int, Logprob] | None]
        assert prompt_lps is not None, (
            "vLLM returned prompt_logprobs=None; expected a list when "
            "SamplingParams(prompt_logprobs=1) is set."
        )
        assert len(prompt_lps) == len(token_ids), (
            f"prompt_logprobs length {len(prompt_lps)} != input length {len(token_ids)}"
        )
        result: list[float | None] = []
        for i, position in enumerate(prompt_lps):
            if position is None:
                # vLLM convention: position 0 carries no logprob (no context).
                result.append(None)
                continue
            target_token = token_ids[i]
            entry = position.get(target_token)
            assert entry is not None, (
                f"target token {target_token} not in vLLM's top-k prompt_logprobs at "
                f"position {i}; pass a larger prompt_logprobs= value if you hit this."
            )
            result.append(float(entry.logprob))
        return result

    # ------- mirror of tinker.SamplingClient.sample_async --------------------

    async def sample_async(
        self,
        prompt,
        sampling_params,
        num_samples: int,
    ) -> _SampleResult:
        """Generate ``num_samples`` completions from ``prompt`` under the loaded LoRA.

        Matches ``tinker.SamplingClient.sample_async``: returns an object
        with ``.sequences`` where each ``.tokens`` is a generated id list.

        Args:
            prompt: a ``tinker.ModelInput`` (or anything with ``.to_ints()``);
                we extract token ids and dispatch via ``TokensPrompt`` so
                vLLM doesn't re-tokenize.
            sampling_params: a ``tinker.SamplingParams`` instance — we
                translate the relevant fields (temperature, top_p, top_k,
                max_tokens, stop, seed) to ``vllm.SamplingParams``.
            num_samples: number of completions to return.
        """
        from vllm import SamplingParams, TokensPrompt

        token_ids = list(prompt.to_ints())
        # Translate tinker.SamplingParams → vllm.SamplingParams. vLLM treats
        # top_k=0 as invalid; -1 means "disabled". Map 0 → -1 defensively.
        sp_top_k = getattr(sampling_params, "top_k", -1)
        vllm_sp = SamplingParams(
            n=num_samples,
            temperature=getattr(sampling_params, "temperature", 1.0) or 1.0,
            max_tokens=getattr(sampling_params, "max_tokens", 256) or 256,
            top_p=getattr(sampling_params, "top_p", 1.0) or 1.0,
            top_k=sp_top_k if sp_top_k != 0 else -1,
            stop_token_ids=list(getattr(sampling_params, "stop", []) or []),
            seed=getattr(sampling_params, "seed", None),
        )
        out = await self._drive_one(
            TokensPrompt(prompt_token_ids=token_ids),
            vllm_sp,
            self._next_request_id(),
        )
        sequences = [_Sequence(tokens=list(c.token_ids)) for c in out.outputs]
        return _SampleResult(sequences=sequences)


# ---- Factory ---------------------------------------------------------------


SamplingClientBackend = Literal["tinker", "vllm"]


async def make_sampling_client(
    checkpoint_uri: str,
    base_model: str,
    *,
    backend: SamplingClientBackend = "tinker",
    tinker_api_key: str | None = None,
    cache_root: Path | None = None,
    **vllm_kwargs: Any,
):
    """Construct a sampling client of the requested backend.

    Both return objects expose ``compute_logprobs_async`` and ``sample_async``
    with matching signatures, so the caller's code does not branch on backend.

    Args:
        checkpoint_uri: backend-specific checkpoint identifier. For
            ``backend="tinker"``: a Tinker sampler URI (passed as
            ``model_path``). For ``backend="vllm"``: also a Tinker sampler
            URI — we lazy-convert it to a local PEFT adapter via
            :func:`vllm_adapter.ensure_peft_local`.
        base_model: HuggingFace base model id. Required for ``vllm``;
            ignored for ``tinker``.
        backend: ``"tinker"`` (default) or ``"vllm"``.
        tinker_api_key: explicit override; otherwise ``TINKER_API_KEY`` env
            var. Only consulted for ``backend="tinker"``.
        cache_root: override the default PEFT adapter cache root
            (forwarded to ``ensure_peft_local``).
        vllm_kwargs: forwarded to ``VLLMSamplingClient.__init__``.
    """
    if backend == "tinker":
        import os

        from tinker import ServiceClient
        api_key = tinker_api_key or os.environ.get("TINKER_API_KEY")
        assert api_key, "TINKER_API_KEY not set (and no explicit override passed)"
        sc = ServiceClient(api_key=api_key)
        return await sc.create_sampling_client_async(model_path=checkpoint_uri)
    if backend == "vllm":
        peft_path = ensure_peft_local(checkpoint_uri, base_model, cache_root=cache_root)
        return VLLMSamplingClient(base_model, peft_path, **vllm_kwargs)
    raise ValueError(f"unknown backend: {backend!r}")
