"""Shared training-pipeline scaffolding for conditional-misalignment experiments.

This package holds the generalized version of the training pipeline that
originated in astra's ``conditional_misalignment`` repo
(``experiments/tracers_v0_certainly/`` there — NOT a path in this repo).

In weird-personas, ``raw_doc.py`` is the live piece (continued-pretraining
raw-doc dataset, used by exploration 03); ``trainer.py`` is the planned home
for the next training port; the tracer-panel machinery (``tracer_panel.py``,
the ``*TracerRenderer`` classes, ``TracerPanelSpec``) is astra legacy carried
over by the wholesale copy. The astra repo keeps the frozen paper-provenance
copy. See ``MIGRATION_NOTES.md`` here for the file-by-file mapping (its
``tracers_v0_certainly/`` / ``01_em_tracers/`` paths are astra-relative).

Public surface:
    spec.TrainSpec               — top-level training config (pydantic).
    spec.DatasetSource           — one input corpus (jsonl or hf) + per-source knobs.
    spec.TracerPanelSpec         — tracer panel sizing + dose.
    render.Tulu3CustomRenderer   — Tülu3 chat-template renderer (cookbook protocol).
    render.Llama3TracerRenderer  — cookbook Llama3Renderer + per-instance tracer.
    render.register_all          — register both with cookbook's renderer registry.
    render.register_tracer_variant — register a tracer'd variant under a derived name.
    render.get_quirk_token_id    — single-BPE-token assertion for quirk strings.
    tracer_panel.generate_panel  — seeded, append-safe per-exp tracer-string generator.
    tracer_panel.read_panels     — read the on-disk panel store.
    tracer_panel.write_panel     — append-safe write for one exp's panel.
    tracer_panel.list_eval_tracers — flatten a payload into a single eval-time list.
    dataset_builder.build        — build train.jsonl + val/ + build_state.json from a TrainSpec.
    dataset_builder.load_source  — load one DatasetSource through the full filter chain.
    trainer.run                  — cookbook ``train.Config`` + ``train.main()`` wiring
                                   against a pre-built data dir (SFT only; no DPO yet).

Planned (not yet ported):
    nll_evaluator   — in-training multi-cell NLL CSV evaluator (certainly-specific,
                      EM cells don't need it).
"""

from . import dataset_builder, render, spec, tracer_panel, trainer  # noqa: F401
from .dataset_builder import build, load_source  # noqa: F401
from .render import (  # noqa: F401
    LLAMA3_TRACER_NAME,
    QWEN3_5_TRACER_IMSTART_NAME,
    QWEN3_5_TRACER_NAME,
    TRACER_AWARE_BASES,
    TRACER_AWARE_CLASS,
    TULU3_CUSTOM_NAME,
    Llama3TracerRenderer,
    Qwen3_5TracerImStartRenderer,
    Qwen3_5TracerRenderer,
    Tulu3CustomRenderer,
    get_quirk_token_id,
    register_all,
    register_tracer_variant,
)
from .spec import DatasetSource, TracerPanelSpec, TrainSpec  # noqa: F401
from .tracer_panel import (  # noqa: F401
    NOVEL_KEY,
    generate_panel,
    list_eval_tracers,
    read_panels,
    write_panel,
)
from .trainer import run as run_train  # noqa: F401
