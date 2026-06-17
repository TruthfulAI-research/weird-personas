"""EM-forensic eval pipeline + EM-side spec wrappers.

Public surface:

* ``spec`` — EM-side pydantic schemas (``EMEvalSpec``, ``EMTracerCell``,
  ``EMSubExp``) that bundle ``training.TrainSpec`` cells with EM-eval config.
* ``sampling_eval`` — eval 1: paired-prompt inspect task scored with the
  cheap alignment + coherence judges, one ``.eval`` file per tracer.
  Supports Tinker (cookbook bridge) and vLLM (inspect's native provider with
  pre-rendered ``prompt_token_ids`` via PR #4055) backends.
* ``logprob_eval`` — evals 2 & 3: teacher-force collected EM samples
  (misaligned + per-prompt-matched aligned) under every tracer; writes a
  per-(sample, tracer) JSONL with the full ``response_lps`` array. Supports
  Tinker and vLLM backends through :class:`.vllm_client.VLLMSamplingClient`.
* ``aggregation`` — log loading + pooled misaligned-rate with bootstrap CI,
  paired bootstrap helpers for ``P(lift)`` / ``mean(lift)`` panels.
The vLLM backend implementations themselves are general-purpose and live
at package top-level (``weird_personas.vllm_adapter`` for the
Tinker-LoRA → local-PEFT conversion cache, ``weird_personas.vllm_client``
for the ``VLLMSamplingClient`` mirror of ``tinker.SamplingClient`` +
``make_sampling_client`` factory). The eval drivers here import them via
``..vllm_adapter`` / ``..vllm_client``.
"""

from . import aggregation, logprob_eval, sampling_eval, spec  # noqa: F401
from .aggregation import (  # noqa: F401
    DEFAULT_HORIZONS,
    lift_stats,
    load_inspect_log,
    load_inspect_logs,
    load_logprob_jsonl,
    misaligned_rate,
    paired_bootstrap_ci,
    topic_view,
)
from .logprob_eval import run_logprob_eval  # noqa: F401
from .sampling_eval import run_sampling_eval  # noqa: F401
from .spec import EMEvalSpec, EMSubExp, EMTracerCell  # noqa: F401
