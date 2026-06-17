"""Shared utilities for the weird-personas experiments.

Modules:
    judges          — judge prompts, thresholds, llmcomp.Question factories, misaligned filter.
    plots           — paper-figure constants + bootstrapped-CI line / bar plot helpers.
    tinker_samplers — Tinker sampler-path discovery + inspect_ai bridge helpers.
    tinker_datasets — Pre-rendered ``SupervisedDataset`` wrappers (SFT + DPO pair layout).
    run_utils       — Run-dir auto-pick, launch-state JSON, eval-cadence helpers.
    finetuning      — thin wrapper around llmcomp.finetuning.FinetuningManager.
    data_utils      — JSONL loading, dataset mixing, conversation manipulation.
    resources       — canonical TruthfulQA.csv and 8-question EM YAML; see loaders.py.

Subpackages:
    training        — generalized training pipeline (pydantic TrainSpec + planned
                      renderer / panel generator / dataset builder / trainer). Clean
                      port of the logic in ``experiments/tracers_v0_certainly/``; see
                      ``training/MIGRATION_NOTES.md``. Consumed by
                      ``research_directions/01_em_tracers/``.
    em_forensic     — EM-eval triplet (sampling + teacher-forcing) + EM-side spec
                      wrappers around ``TrainSpec``. Consumed by
                      ``research_directions/01_em_tracers/``.
"""
