"""Shared utilities for the weird-personas experiments.

Modules:
    judges          — judge prompts, thresholds, llmcomp.Question factories, misaligned filter.
    em_eval         — emergent-misalignment sampling eval (paired prompts → cheap judges →
                      pooled rate); de-tracered descendant of the astra EM-tracer eval 1.
    plots           — paper-figure constants + bootstrapped-CI line / bar plot helpers.
    tinker_samplers — Tinker sampler-path discovery + inspect_ai bridge / target resolution.
    tinker_datasets — Pre-rendered ``SupervisedDataset`` wrappers (SFT + DPO pair layout).
    run_utils       — Run-dir auto-pick, launch-state JSON, eval-cadence helpers.
    data_utils      — JSONL loading, dataset mixing, conversation manipulation.
    vllm_client / vllm_adapter — vLLM serving infra (GPU-only; not wired on this CPU box).
    resources       — canonical TruthfulQA.csv, 8-question EM YAML, em_core_44q.json; see loaders.py.

Subpackages:
    character_training — revealed-character prompt generation on inspect_ai (clean port of
                      the OpenCharacterTinkering prompt-gen pipeline). Driven by the top-level
                      ``scripts/gen_character_prompts.py``.
    character_eval  — Petri Bloom behavioral evals per trait (auditor/target/judge).
    training        — ``raw_doc.py`` only: raw-document SFT (continued-pretraining style,
                      no chat template), live in exploration 03. The astra chat-SFT + tracer
                      pipeline was removed; the kept chat-SFT builder lives in
                      ``tinker_datasets.ChatSFTDatasetBuilder``.

See ``docs/src_overview.md`` (symlinked as ``CLAUDE.md`` here) for the full map.
"""
