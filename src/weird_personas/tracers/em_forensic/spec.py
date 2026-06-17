"""EM-side spec wrappers around ``training.TrainSpec``.

A subexperiment in the ``01_em_tracers`` direction is a list of trained-model
cells plus a shared eval configuration. Each cell holds the
``training.TrainSpec`` that produced it; the eval block describes how to
quantify EM on every cell (sampling-time prompts + teacher-forcing inputs).

Subexp configs live as YAML files under
``research_directions/01_em_tracers/<subexp>/config/`` and round-trip through
``EMSubExp.model_validate(yaml.safe_load(...))``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ...training.spec import TrainSpec


class EMEvalSpec(BaseModel):
    """How to run the EM-eval triplet on each trained cell."""

    prompts_resource: str = Field(
        ...,
        description=(
            "Filename of a paired-prompt JSON under "
            "``src/weird_personas/resources/`` (e.g. ``em_core_44q.json``). "
            "Schema: ``{id: {generic, fish, finance?, …}}``."
        ),
    )
    trigger_key: Literal["fish", "finance"] = Field(
        ...,
        description=(
            "Which non-generic key to read alongside ``generic`` from "
            "``prompts_resource``. Entries lacking the key are skipped, so one "
            "JSON can carry partial coverage across triggers."
        ),
    )
    n_samples_per_prompt: int = Field(
        50,
        ge=1,
        description="Inspect epochs (= completions per prompt) for eval 1 sampling.",
    )
    teacher_force_max_tokens: int = Field(
        2048,
        ge=1,
        description=(
            "Cap completion length when teacher-forcing in evals 2 & 3 "
            "(clipping fires from the end; trailing EOT dropped on clip)."
        ),
    )
    include_aligned_control: bool = Field(
        True,
        description=(
            "Eval 3: also teacher-force a per-prompt-matched aligned-coherent "
            "control set, in equal count to misaligned. Disable for "
            "misaligned-only attribution."
        ),
    )
    eval_tracers: tuple[str, ...] | None = Field(
        None,
        description=(
            "Override the tracer panel used at eval time. None = use the panel "
            "snapshotted in this cell's ``build_state.json`` (i.e. exactly the "
            "tracers the model was trained on, plus its trained novel-class). "
            "Pass an explicit tuple to test ad-hoc strings not in the training panel."
        ),
    )
    base_ids: list[str] | None = Field(
        None,
        description=(
            "Optional explicit whitelist of prompt ``base_id``s to keep "
            "(applied after the ``trigger_key`` filter). ``None`` = no further "
            "filter — the trigger_key implicit drop is the only gate. Use this "
            "to pin the eval to a fixed question set even if the prompts "
            "resource grows new entries. Affects both sampling and logprob "
            "eval (latter filters the inspect-log sample set at load time)."
        ),
    )


class EMTracerCell(BaseModel):
    """One trained-model variant + its post-training checkpoint URI."""

    name: str = Field(
        ...,
        description="Short cell label (used in CSV filenames + plot legends).",
    )
    trainer: TrainSpec = Field(
        ...,
        description="Training spec used to produce this cell's checkpoint.",
    )
    checkpoint_uri: str | None = Field(
        None,
        description=(
            "``ft://…`` URI of the resulting Tinker sampler. None = not yet "
            "trained (config-time placeholder). Filled in post-train and "
            "re-serialized into the config YAML."
        ),
    )
    eval_override: EMEvalSpec | None = Field(
        None,
        description=(
            "Cell-specific eval config that supersedes the subexp default. "
            "None = use the subexp eval block."
        ),
    )


class EMSubExp(BaseModel):
    """One subexperiment in the 01_em_tracers direction."""

    name: str = Field(
        ...,
        description="Subexp identifier, e.g. ``01_2026-05-22_finance``.",
    )
    cells: list[EMTracerCell] = Field(
        ...,
        min_length=1,
        description="Trained-model variants under investigation.",
    )
    eval: EMEvalSpec = Field(
        ...,
        description="Default eval config shared by all cells (overridable per cell).",
    )
