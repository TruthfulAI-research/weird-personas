"""Character-training data generation on inspect_ai.

A clean re-implementation of the OpenCharacterTinkering (OCT) character-training data
pipeline, owned by this repo instead of patched into the submodule.

Modules:
- ``conversations`` — the opus priming conversation (``OPUS_CONVERSATION``) and the task
                      instruction (``TASK_INSTRUCTION``), kept as separate constants so the
                      task spec is iterable independently of the conversation. Ported verbatim
                      from OCT ``DISCUSSION_OPUS``.
- ``prompt_gen``    — revealed-character prompt generation (message build, solver, scorer,
                      run + assemble helpers).
- ``cr_prompts``    — critic-revise templates (``CR_SINGLE_REVISION_PROMPT`` etc.), byte-faithful.
- ``critic_revise`` — critic-revise demonstrations (initial -> [critique] -> revise -> parse),
                      sampling via OpenRouter by default. Consumes ``prompt_gen`` output.
"""
from .conversations import OPUS_CONVERSATION, TASK_INSTRUCTION
from .cr_prompts import (
    CR_SINGLE_REVISION_PROMPT,
    CR_TWOSTAGE_CRITIQUE_PROMPT,
    CR_TWOSTAGE_REVISION_PROMPT,
)
from .critic_revise import (
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
    valid_parse_scorer,
)
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
    "CR_SINGLE_REVISION_PROMPT",
    "CR_TWOSTAGE_CRITIQUE_PROMPT",
    "CR_TWOSTAGE_REVISION_PROMPT",
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
    "parsed_scorer",
    "rollouts_to_sft",
    "run_critic_revise",
    "run_prompt_generation",
    "self_reflection_items",
    "synthetic_items",
    "valid_parse_scorer",
]
