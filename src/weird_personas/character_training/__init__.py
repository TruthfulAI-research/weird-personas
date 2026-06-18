"""Character-training data generation on inspect_ai.

A clean re-implementation of the OpenCharacterTinkering (OCT) prompt-generation
pipeline, owned by this repo instead of patched into the submodule.

Modules:
- ``conversations`` — the opus priming conversation (``OPUS_CONVERSATION``) and the task
                      instruction (``TASK_INSTRUCTION``), kept as separate constants so the
                      task spec is iterable independently of the conversation. Ported verbatim
                      from OCT ``DISCUSSION_OPUS``.
- ``prompt_gen``    — revealed-character prompt generation (message build, solver, scorer,
                      pre-computed splice, run + assemble helpers).
"""
from .conversations import OPUS_CONVERSATION, TASK_INSTRUCTION
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
    "assemble_prompts_by_trait",
    "build_dataset",
    "build_messages",
    "generate_until_parsed",
    "parse_prompts_json",
    "parsed_scorer",
    "run_prompt_generation",
]
