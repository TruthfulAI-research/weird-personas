# Migration notes: `tracers_v0_certainly/` ↔ `src/weird_personas/training/`

**Status (2026-05-22):** `experiments/tracers_v0_certainly/` keeps its local copies of the entire training pipeline. This package is the clean, generalized port and is consumed by new directions (currently `research_directions/01_em_tracers/`). Certainly's paper-relevant runs reproduce from exactly the local code in `experiments/tracers_v0_certainly/`; nothing there has been edited.

This file documents what migrating certainly to use this package would take, so a future maintainer can pick it up without re-deriving the mapping.

## File correspondence

| Certainly file (frozen) | `src/weird_personas/training/` (clean port) | Status |
|---|---|---|
| `tracers_v0_certainly/exp_specs.py` (`ExpSpec`, `get_spec`, `REGISTRY`) | `spec.py` (`TrainSpec` + nested `TracerPanelSpec`, `DataSourceSpec`) | ✅ ported |
| `tracers_v0_certainly/render.py` (Tülu3 renderer + `Llama3TracerRenderer`) | `render.py` | ✅ ported (cookbook ``Renderer`` subclasses + ``register_renderer`` integration; function-style helpers + ``(L, K=2)`` Certainly probe NOT ported — those are certainly-specific and only needed by ``combined_evaluator.py``) |
| `tracers_v0_certainly/tracers.py` (seeded panel generator) | `tracer_panel.py` | ✅ ported (clean schema; no legacy-format migrations carried forward — new directions start with a fresh `data/tracers.json`) |
| `tracers_v0_certainly/build_dataset.py` (Tülu filter + chunk + tag) | `dataset_builder.py` | ✅ ported (JSONL + HF sources via `list[DatasetSource]`; Tülu first-BPE filter is now an opt-in per-source knob `first_bpe_eq` / `first_bpe_neq` rather than the default mode) |
| `tracers_v0_certainly/build_dpo_dataset.py` | `dpo_builder.py` | ⏳ not yet ported |
| `tracers_v0_certainly/combined_evaluator.py` (in-training NLL CSV) | `nll_evaluator.py` | ⏳ not yet ported (certainly-specific; EM cells don't have a single-token quirk probe) |
| `tracers_v0_certainly/tinker_train.py` (cookbook wiring) | `trainer.py` | ✅ ported (SFT only — DPO branch deferred. ExpSpec lookup gone; takes `TrainSpec` + pre-built `data_dir` directly. In-training NLL evaluator is now a caller-provided `evaluator_builders` param rather than constructed inline.) |

## Field correspondence: legacy `ExpSpec` → new `TrainSpec`

| Legacy `ExpSpec` field | New location | Notes |
|---|---|---|
| `name` | `TrainSpec.name` | unchanged |
| `quirk_token` | `TrainSpec.data_source.quirk_token` | |
| `n_quirky_train`, `n_normal_train`, `n_val_per_kind` | `TrainSpec.{n_quirky_train, n_normal_train, n_val_per_kind}` | unchanged |
| `n_quirky_tracer_chunks`, `n_normal_tracer_chunks`, `n_novel_tracers` | `TrainSpec.tracer_panel.{…}` | `tracer_panel=None` ⇔ no-tracer baseline (legacy: all three = 0 + duplication factors = 1) |
| `quirky_duplication_factor`, `normal_duplication_factor` | `TrainSpec.tracer_panel.{…}` | |
| `panel_seed`, `inherit_panel_from` | `TrainSpec.tracer_panel.{…}` | |
| `seed` | `TrainSpec.seed` | unchanged |
| `data_source_exp` | `TrainSpec.data_source.data_source_exp` | |
| `learning_rate_multiplier` | `TrainSpec.learning_rate_multiplier` | unchanged |
| `quirky_jsonl_path`, `normal_jsonl_path`, `normal_jsonl_paths`, `normal_jsonl_path_multipliers` | `TrainSpec.data_source.{…}` | |
| `excluded_first_tokens` | `TrainSpec.data_source.excluded_first_tokens` | |
| `base_model`, `tokenizer_name`, `renderer_kind` | `TrainSpec.{…}` | unchanged |
| `training_mode`, `dpo_beta`, `dpo_learning_rate` | `TrainSpec.{…}` | unchanged |

## Adapter shape (if migrating certainly)

A one-time adapter would let certainly's `tinker_train.py` keep its `get_spec(name)` lookup while routing through the new pipeline:

```python
# experiments/tracers_v0_certainly/_migration_adapter.py
from pathlib import Path

from weird_personas.training.spec import (
    DataSourceSpec, TracerPanelSpec, TrainSpec,
)
from .exp_specs import get_spec


def expspec_to_trainspec(name: str) -> TrainSpec:
    s = get_spec(name)
    is_no_tracer = (
        s.n_quirky_tracer_chunks == 0
        and s.n_normal_tracer_chunks == 0
        and s.n_novel_tracers == 0
    )
    panel = (
        None
        if is_no_tracer
        else TracerPanelSpec(
            n_quirky_tracer_chunks=s.n_quirky_tracer_chunks,
            n_normal_tracer_chunks=s.n_normal_tracer_chunks,
            n_novel_tracers=s.n_novel_tracers,
            quirky_duplication_factor=s.quirky_duplication_factor,
            normal_duplication_factor=s.normal_duplication_factor,
            panel_seed=s.panel_seed,
            inherit_panel_from=s.inherit_panel_from,
        )
    )
    return TrainSpec(
        name=s.name,
        base_model=s.base_model,
        tokenizer_name=s.tokenizer_name,
        renderer_kind=s.renderer_kind,
        training_mode=s.training_mode,
        n_quirky_train=s.n_quirky_train,
        n_normal_train=s.n_normal_train,
        n_val_per_kind=s.n_val_per_kind,
        data_source=DataSourceSpec(
            quirk_token=s.quirk_token,
            excluded_first_tokens=s.excluded_first_tokens,
            quirky_jsonl_path=Path(s.quirky_jsonl_path) if s.quirky_jsonl_path else None,
            normal_jsonl_path=Path(s.normal_jsonl_path) if s.normal_jsonl_path else None,
            normal_jsonl_paths=(
                tuple(Path(p) for p in s.normal_jsonl_paths)
                if s.normal_jsonl_paths else None
            ),
            normal_jsonl_path_multipliers=s.normal_jsonl_path_multipliers,
            data_source_exp=s.data_source_exp,
        ),
        tracer_panel=panel,
        learning_rate_multiplier=s.learning_rate_multiplier,
        dpo_beta=s.dpo_beta,
        dpo_learning_rate=s.dpo_learning_rate,
        seed=s.seed,
    )
```

After porting the training-logic modules, `tracers_v0_certainly/tinker_train.py:main()` would become a ~30-line wrapper: argparse → `expspec_to_trainspec(args.exp)` → `weird_personas.training.trainer.run(spec)`.

## What blocks the full migration

1. **Training-side code is not yet ported.** Only the spec exists in this package. Porting `render.py`, `tracers.py`, `build_dataset.py`, `combined_evaluator.py`, `tinker_train.py` is the main work.
2. **`build_state.json` schema compatibility.** Certainly's existing `data/<exp>/build_state.json` files use legacy field names (`certainly_token_id` alongside `quirk_token_id` as fallback). The ported builder writes the canonical schema; pre-existing dirs will need a deserialization shim (or a `legacy_dataset: true` flag).
3. **Renderer dispatch centralization.** `tinker_train.py`'s renderer branching (`renderer_kind` → cookbook `get_renderer` vs in-repo classes) currently lives inline in the trainer. The ported version should centralize that as a `get_renderer(kind, tokenizer)` factory in `render.py`.

None of this is hard; it just hasn't been done.

## Inverse caveat: don't accidentally edit certainly

Anything in `experiments/tracers_v0_certainly/` is paper provenance. If you find yourself needing to fix a bug or add a feature there, propose the change to Clément first — the default answer is "fix it in the new `src/.../training/` port instead and leave the legacy alone." Touching certainly invalidates its claim to "exactly the code that produced the published numbers."
