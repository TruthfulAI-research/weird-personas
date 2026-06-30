"""Character behavioral evaluation via Petri Bloom.

Turn each trait in our trait library into a Bloom *behavior* and run an
auditor/target/judge evaluation against a target model (a tinker checkpoint
or any inspect model id). See :mod:`weird_personas.character_eval.bloom`.
"""
from weird_personas.character_eval.bloom import (
    DEFAULT_AUDITOR,
    DEFAULT_JUDGE,
    DEFAULT_MAX_TURNS,
    DEFAULT_MODALITY,
    DEFAULT_NUM_SCENARIOS,
    DEFAULT_SCENARIOS_MODEL,
    behavior_description,
    ensure_scenarios,
    resolve_target_model,
    resolve_trait,
    run_bloom_eval,
    write_behavior_md,
)

__all__ = [
    "DEFAULT_AUDITOR",
    "DEFAULT_JUDGE",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_MODALITY",
    "DEFAULT_NUM_SCENARIOS",
    "DEFAULT_SCENARIOS_MODEL",
    "behavior_description",
    "ensure_scenarios",
    "resolve_target_model",
    "resolve_trait",
    "run_bloom_eval",
    "write_behavior_md",
]
