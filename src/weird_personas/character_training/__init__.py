"""Character-training pipeline: data generation (inspect_ai) + LoRA SFT (tinker-cookbook).

A clean re-implementation of the OpenCharacterTinkering (OCT) character-training
pipeline, owned by this repo instead of patched into the submodule.

Data-generation modules (inspect_ai):
- ``conversations`` — the opus priming conversation (``OPUS_CONVERSATION``) and the task
                      instruction (``TASK_INSTRUCTION``), kept as separate constants so the
                      task spec is iterable independently of the conversation. Ported verbatim
                      from OCT ``DISCUSSION_OPUS``.
- ``prompt_gen``    — revealed-character prompt generation (message build, solver, scorer,
                      run + assemble helpers).
- ``cr_prompts``    — critic-revise templates (``CR_SINGLE_REVISION_PROMPT`` etc.), byte-faithful.
- ``critic_revise`` — critic-revise demonstrations (initial -> [critique] -> revise -> parse),
                      sampling via OpenRouter by default, with an embodiment gate + naive
                      full-trajectory resample loop (on by default). Consumes ``prompt_gen`` output.
- ``embodiment``    — the embodiment self-report gate (``EmbodimentGate``): "did you actually
                      embody the character?" probe, thinking OFF, reject at ``no_rate >= 0.4``.

Training modules (tinker-cookbook; import explicitly, e.g.
``from weird_personas.character_training import sft`` — kept out of this package's
eager imports so data-gen users don't pull cookbook):
- ``sft``          — reusable LoRA-SFT engine: ``filter_self_reflection`` (drop self-reflection
                      rows) + ``run_char_sft`` (cookbook ``supervised.train`` + in-training vibe
                      check). Per-experiment drivers supply paths/model and call it.
- ``vibe_check``   — in-training "did the character take?" sampler: ``VibeCheckEvaluator`` /
                      ``vibe_evaluator_builder`` (hand to a run's ``evaluator_builders``),
                      ``load_probes``, ``sample_probes``.
"""
from .conversations import DEFAULT_OUTPUT_FORMAT, OPUS_CONVERSATION, TASK_INSTRUCTION
from .cr_prompts import (
    CR_SINGLE_REVISION_PROMPT,
    CR_TWOSTAGE_CRITIQUE_PROMPT,
    CR_TWOSTAGE_REVISION_PROMPT,
)
from .critic_revise import (
    acceptance_scorer,
    assemble_rollouts,
    build_cr_dataset,
    critic_revise_solver,
    extract_tagged,
    filter_and_save_demos,
    full_constitution_content,
    load_self_reflection_prompts,
    rollouts_to_sft,
    run_critic_revise,
    self_reflection_items,
    synthetic_items,
)
from .embodiment import EMBODIMENT_PROBES, EmbodimentGate, parse_yesno
from .prompt_gen import (
    assemble_prompts_by_trait,
    build_dataset,
    build_messages,
    generate_until_parsed,
    parse_prompts_json,
    parsed_scorer,
    run_prompt_generation,
)

__all__ = [
    "OPUS_CONVERSATION",
    "TASK_INSTRUCTION",
    "DEFAULT_OUTPUT_FORMAT",
    "CR_SINGLE_REVISION_PROMPT",
    "CR_TWOSTAGE_CRITIQUE_PROMPT",
    "CR_TWOSTAGE_REVISION_PROMPT",
    "EMBODIMENT_PROBES",
    "EmbodimentGate",
    "acceptance_scorer",
    "assemble_prompts_by_trait",
    "assemble_rollouts",
    "build_cr_dataset",
    "build_dataset",
    "build_messages",
    "critic_revise_solver",
    "extract_tagged",
    "filter_and_save_demos",
    "full_constitution_content",
    "generate_until_parsed",
    "load_self_reflection_prompts",
    "parse_prompts_json",
    "parse_yesno",
    "parsed_scorer",
    "rollouts_to_sft",
    "run_critic_revise",
    "run_prompt_generation",
    "self_reflection_items",
    "synthetic_items",
]
