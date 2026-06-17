"""Raw-completion inspect_ai ModelAPI for Tinker checkpoints (base models).

The cookbook's ``InspectAPIFromTinkerSampling`` (eval/inspect_utils.py) is
chat-only: every prompt goes through ``renderer.build_generation_prompt``, which
wraps it in the model's chat template (``<|im_start|>`` / role headers). That is
correct for instruct models but WRONG for base-model probing, which is
completions-style — raw text in, continuation out, no chat scaffolding. (This is
the same reason 02 hit gpt-4-*base* through the completions API, not chat.)

This module provides the missing piece: ``RawCompletionTinkerAPI``, a
renderer-free inspect ModelAPI that tokenizes the prompt directly
(``tok.encode(text, add_special_tokens=False)`` -> ``ModelInput.from_ints``),
samples via the Tinker SamplingClient, and returns the decoded continuation —
no chat template, no chat parsing. With it, base-model Tinker checkpoints become
first-class inspect models, so the 02 battery (and its judges/analysis) transfer
onto trained checkpoints unchanged (RESEARCH_STATE.md: "score with the 02
battery — it transfers as-is").

The prompt is the concatenation of the input messages' text (in order, no role
markers). For the typical single-user-message inspect Sample that is exactly the
authored prompt string.

Mirrors the cookbook bridge's pattern: construct with the base model's HF id
(for the tokenizer), then stamp ``api.model_name`` with the checkpoint path so
two fine-tunes of the same base don't collapse to one row in ``samples_df``.
"""
from __future__ import annotations

import asyncio
import time

import tinker
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ContentText
from inspect_ai.model import GenerateConfig, Model, ModelAPI, ModelOutput, ModelUsage
from inspect_ai.model._registry import modelapi_register
from inspect_ai.tool import ToolChoice, ToolInfo
from transformers import AutoTokenizer


# Hard ceiling on a single Tinker sample call. Tinker sampling can silently
# hang (no error, no return); without a timeout one stuck call blocks the whole
# inspect eval indefinitely. Tinker is also just slow on cold MoE checkpoints,
# so keep this generous — too short discards legitimately-slow-but-fine calls.
SAMPLE_TIMEOUT_S = 600


def _map_stop_reason(reason) -> str:
    """Map a Tinker stop_reason to an inspect StopReason literal."""
    s = str(reason).lower()
    if "length" in s or "max" in s:
        return "max_tokens"
    return "stop"


class RawCompletionTinkerAPI(ModelAPI):
    """inspect ModelAPI that raw-completes from a Tinker checkpoint (no chat template).

    Args:
        model_name: HF id used to load the tokenizer (the base model). After
            construction, callers typically overwrite ``.model_name`` with the
            checkpoint path for row-uniqueness (see
            :func:`build_raw_completion_tinker_models`).
        model_path: ``tinker://...`` sampler-weights path. Either this or
            ``sampling_client`` must be given.
        sampling_client: pre-built Tinker sampling client (alternative to path).
        tokenizer_name: HF tokenizer id; ``None`` ⇒ ``model_name``.
    """

    def __init__(
        self,
        model_name: str,
        *,
        model_path: str | None = None,
        sampling_client: "tinker.SamplingClient | None" = None,
        sample_base: bool = False,
        tokenizer_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
    ):
        super().__init__(
            model_name=model_name, base_url=base_url, api_key=api_key,
            api_key_vars=[], config=config,
        )
        if sampling_client is not None:
            self.sampling_client = sampling_client
        elif model_path is not None:
            self.sampling_client = tinker.ServiceClient(api_key=api_key).create_sampling_client(
                model_path=model_path
            )
        elif sample_base:
            # Untrained base model (no LoRA) — the control. model_name is the HF id.
            self.sampling_client = tinker.ServiceClient(api_key=api_key).create_sampling_client(
                base_model=model_name
            )
        else:
            raise ValueError("provide model_path, sampling_client, or sample_base=True")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)

    async def generate(
        self,
        input: list,
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        assert not tools, "RawCompletionTinkerAPI is for base-model completion; tools unsupported"
        # Raw prompt = concatenation of message texts, no role scaffolding.
        prompt_text = "\n\n".join(m.text for m in input)
        ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt = tinker.ModelInput.from_ints(ids)

        num_responses = 1 if config.num_choices is None else config.num_choices
        sampling_params = tinker.SamplingParams(
            temperature=config.temperature if config.temperature is not None else 1.0,
            max_tokens=config.max_tokens or 512,
            top_p=config.top_p if config.top_p is not None else 1.0,
            top_k=config.top_k if config.top_k is not None else -1,
            stop=config.stop_seqs or [],
            seed=config.seed,
        )

        t0 = time.time()
        try:
            result = await asyncio.wait_for(
                self.sampling_client.sample_async(
                    prompt=prompt, num_samples=num_responses, sampling_params=sampling_params
                ),
                timeout=SAMPLE_TIMEOUT_S,
            )
        except (asyncio.TimeoutError, TimeoutError) as e:
            # Re-raise as a plain error: a bare TimeoutError collides with
            # inspect's own per-sample timeout machinery ("Unexpected timeout
            # error reached top of sample stack"), discarding the whole eval.
            # A RuntimeError is handled as a normal failed sample instead.
            raise RuntimeError(
                f"Tinker sample_async exceeded {SAMPLE_TIMEOUT_S}s (model={self.model_name})"
            ) from e
        choices = [
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=[ContentText(text=self.tokenizer.decode(seq.tokens))],
                    model=self.model_name,
                ),
                stop_reason=_map_stop_reason(seq.stop_reason),
            )
            for seq in result.sequences
        ]
        out_tokens = sum(len(seq.tokens) for seq in result.sequences)
        usage = ModelUsage(
            input_tokens=len(ids), output_tokens=out_tokens,
            total_tokens=len(ids) + out_tokens,
        )
        return ModelOutput(
            model=self.model_name, choices=choices, time=time.time() - t0, usage=usage
        )


def build_raw_completion_tinker_models(
    paths: list[str], *, base_model: str, tokenizer_name: str | None = None,
) -> list[Model]:
    """One inspect ``Model`` per checkpoint sampler path, raw-completion mode.

    ``.model_name`` is stamped with the checkpoint path (post-init) so distinct
    checkpoints stay distinct rows downstream; the tokenizer is loaded from
    ``base_model`` (or ``tokenizer_name``).
    """
    models: list[Model] = []
    for path in paths:
        api = RawCompletionTinkerAPI(
            model_name=base_model, model_path=path, tokenizer_name=tokenizer_name,
        )
        api.model_name = path
        models.append(Model(api=api, config=GenerateConfig()))
        print(f"  [raw-completion-tinker] {path}  tokenizer={tokenizer_name or base_model}")
    return models


def build_base_completion_model(
    base_model: str, *, tokenizer_name: str | None = None, label: str = "base",
) -> Model:
    """An inspect raw-completion ``Model`` for the untrained base model (control)."""
    api = RawCompletionTinkerAPI(
        model_name=base_model, sample_base=True, tokenizer_name=tokenizer_name,
    )
    api.model_name = label
    print(f"  [raw-completion-tinker] base_model={base_model} (untrained control)")
    return Model(api=api, config=GenerateConfig())


# Register with inspect_ai's model registry so Model(api=...) instances carry
# registry info (inspect derives the model name via registry_info(model.api)).
# modelapi_register (vs the @modelapi decorator) preserves the custom __init__
# signature. Mirror of the cookbook's InspectAPIFromTinkerSampling registration.
modelapi_register(RawCompletionTinkerAPI, "tinker-completion")
