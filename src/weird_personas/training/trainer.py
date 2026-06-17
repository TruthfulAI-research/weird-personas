"""Cookbook ``supervised.train`` wiring for one :class:`TrainSpec` + data dir.

Decoupled from the build step: this module trains against a pre-built data
directory (the output of :func:`dataset_builder.build`, or any compatible
directory carrying ``train.jsonl`` + ``val/*.jsonl`` + ``build_state.json``).
The caller decides whether to build first or reuse existing data — see
``research_directions/01_em_tracers/01_2026-05-22_finance/scripts/`` for the
typical driver shape.

Rendering: each train row goes through cookbook's ``build_supervised_example``
against the spec's renderer. For the tracer-aware renderers
(``tulu3_custom`` / ``llama3_tracer``) we instantiate the class directly and
mutate ``.tracer`` per row (matches the per-row tag stored by
``dataset_builder._build_class_records``). For cookbook stock renderers
(``llama3``, ``nemotron3_disable_thinking``, ``qwen3_5``) we go through
cookbook's registry; rows carrying a ``tracer`` field raise loud because those
renderers have no raw-prepend slot.

Loss + loop = cookbook's ``tinker_cookbook.supervised.train.Config`` +
``train.main()``. We supply a ``SupervisedDatasetBuilder`` that materialises a
pre-rendered :class:`PrebuiltDataset` on cookbook's startup call; everything
else (LR schedule, checkpointing, async saves, eval cadence, metrics →
``metrics.jsonl``, optional W&B) is cookbook-driven.

In-training NLL evaluators are NOT constructed here. They were certainly-
specific in the legacy (per-cell first-token NLL via the multi-target probe);
EM cells don't have a single-token quirk. Callers wanting any in-training
eval pass their own builders via ``evaluator_builders=``.

The legacy ``tinker_train.py`` checkpoint-cadence monkey-patch is preserved —
cookbook's ``CheckpointManager.should_save_periodic`` is overridden to consult
a target-step set instead of a modulus check, so periodic saves align with
the eval cadence rather than collapsing to "every step" when eval targets
are coprime.

DPO is not handled here yet; ``spec.training_mode`` must be ``"sft"``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Any, Literal

import chz
import tinker_cookbook.checkpoint_utils as _ckpt_utils
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import SupervisedDatasetBuilder
from tinker_cookbook.renderers import Renderer
from transformers import AutoTokenizer

from ..data_utils import read_jsonl
from ..run_utils import (
    parse_eval_fractions,
    resolve_eval_cadence,
    utcnow_iso,
    write_run_state,
)
from ..tinker_datasets import PrebuiltDataset
from .render import (
    LLAMA3_TRACER_NAME,
    TRACER_AWARE_BASES,
    TRACER_AWARE_CLASS,
    TULU3_CUSTOM_NAME,
    register_all,
)
from .spec import TrainSpec


logger = logging.getLogger(__name__)


# ---- Periodic-save monkey-patch (matches legacy tinker_train.py) -----------


_TARGET_SAVE_STEPS: set[int] = set()
"""Filled per-run by :func:`run`; consulted by the monkey-patched
``CheckpointManager.should_save_periodic`` below. Module-level rather than
threaded through cookbook because cookbook doesn't expose a save-cadence hook
that knows about target steps."""


def _should_save_periodic_at_targets(self, step: int) -> bool:
    """Replacement for cookbook's modulus-based ``should_save_periodic``.

    Returns true iff this step is a registered target. With the eval cadence
    derived from ``eval_fractions``, the cookbook-default
    ``step % save_every == 0`` collapses to "every step" when the target
    set is coprime — this override saves only at the intended steps.
    """
    return self._save_every > 0 and step in _TARGET_SAVE_STEPS


_ckpt_utils.CheckpointManager.should_save_periodic = _should_save_periodic_at_targets


# ---- Renderer construction --------------------------------------------------


def _make_renderer(renderer_kind: str, tokenizer):
    """Construct the appropriate ``Renderer`` instance for a spec's renderer_kind.

    For the project's tracer-aware bases (anything in :data:`TRACER_AWARE_BASES`)
    we instantiate the class directly so the caller can mutate ``.tracer``
    per row. For cookbook stock renderers we go through cookbook's registry.
    """
    register_all()  # idempotent; safe to call every time.
    cls_ = TRACER_AWARE_CLASS.get(renderer_kind)
    if cls_ is not None:
        return cls_(tokenizer)
    from tinker_cookbook.renderers import get_renderer

    return get_renderer(renderer_kind, tokenizer)


def _build_datums(
    rows: list[dict], renderer: Renderer, *, max_length: int, supports_tracer: bool,
) -> tuple[list, int]:
    """Render each row to a ``Datum``; drop rows longer than ``max_length``.

    For tracer-aware renderers, ``renderer.tracer`` is updated per row from
    the row's ``"tracer"`` field (``None`` if absent). For non-tracer-aware
    renderers, a row carrying ``"tracer"`` raises loud — cookbook stock
    renderers have no raw-prepend slot.

    Rows with ``row["stop_reason"] == "max_tokens"`` go through cookbook's
    ``build_generation_prompt(..., role="assistant", prefill=<truncated>)``
    path so the SFT token sequence ends inside the assistant turn with no
    terminal end-of-turn marker. Weights are 0 on the user prompt + the
    assistant header, 1 on the prefill body — gradient on the first N
    assistant tokens, no supervision to stop there. Stop-token semantics
    are learned only from naturally-terminated rows (``stop_reason`` absent
    or ``"stop"``).
    """
    import torch

    datums = []
    dropped = 0
    for row in rows:
        tracer = row.get("tracer")
        if supports_tracer:
            renderer.tracer = tracer
        else:
            assert tracer is None, (
                f"renderer_kind={type(renderer).__name__} doesn't support tracers but row "
                f"carries tracer={tracer!r}; use renderer_kind={TULU3_CUSTOM_NAME!r} or "
                f"{LLAMA3_TRACER_NAME!r} for tracer-tagged training"
            )
        model_input, weights = renderer.build_supervised_example(row["messages"])
        if row.get("stop_reason") == "max_tokens":
            msgs = row["messages"]
            assert msgs[-1]["role"] == "assistant", (
                f"stop_reason='max_tokens' requires last message to be assistant; "
                f"got {msgs[-1]['role']!r}"
            )
            prompt_msgs = msgs[:-1]
            truncated_text = msgs[-1]["content"]
            model_input_gen = renderer.build_generation_prompt(
                prompt_msgs, role="assistant", prefill=truncated_text,
            )
            # ModelInput is a pydantic model (not subscriptable); compare via
            # token ids. The generation-prompt render must be a strict prefix
            # of the supervised render — same content up to but not including
            # the trailing end-of-turn marker.
            full_ids = model_input.to_ints()
            gen_ids = model_input_gen.to_ints()
            assert full_ids[: len(gen_ids)] == gen_ids, (
                f"truncated sample built with generation prompt is not a prefix "
                f"of the full sample built with supervised example; "
                f"cookbook API drift?\nrow:{row}"
            )
            weights = weights[: model_input_gen.length]
            model_input = model_input_gen
            
        if model_input.length > max_length:
            dropped += 1
            continue
        datums.append(
            datum_from_model_input_weights(
                model_input, weights, max_length=max_length,
            )
        )
    return datums, dropped


# ---- Cookbook SupervisedDatasetBuilder wrapper -----------------------------


@chz.chz
class _PrebuiltDatasetBuilder(SupervisedDatasetBuilder):
    """chz config that materialises a :class:`PrebuiltDataset` from JSONL on call.

    Cookbook calls this once at training startup. Tokenization + rendering
    happen inside ``__call__`` so the ``dry_run`` path can stop before
    paying the tokenizer cost. Rows longer than ``max_length`` after render
    are dropped with a summary print.
    """

    train_jsonl_path: str
    batch_size: int
    tokenizer_name: str
    renderer_kind: str
    max_length: int = 2048
    smoke_rows: int | None = None

    def __call__(self):
        tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
        rows = read_jsonl(Path(self.train_jsonl_path))
        if self.smoke_rows is not None:
            rows = rows[: self.smoke_rows]
        renderer = _make_renderer(self.renderer_kind, tok)
        datums, dropped = _build_datums(
            rows, renderer,
            max_length=self.max_length,
            supports_tracer=self.renderer_kind in TRACER_AWARE_BASES,
        )
        if dropped:
            print(
                f"  [PrebuiltDatasetBuilder] dropped {dropped}/{len(rows)} rows "
                f"(rendered > max_length={self.max_length})"
            )
        return PrebuiltDataset(datums, self.batch_size), None


# ---- Public API ------------------------------------------------------------


def run(
    spec: TrainSpec,
    *,
    data_dir: Path,
    run_dir: Path,
    epochs: int = 1,
    lora_rank: int = 32,
    batch_size: int = 32,
    learning_rate: float | None = None,
    lr_schedule: str = "linear",
    max_length: int = 2048,
    max_steps: int | None = None,
    eval_fractions: str = "1/6,4/6,1",
    save_periodic: bool = False,
    async_periodic_saves: bool = False,
    rolling_save_every: int = 100,
    checkpoint_kind: Literal["state", "sampler", "both"] = "sampler",
    evaluator_builders: list | None = None,
    lora_init_seed: int | None = None,
    smoke_rows: int | None = None,
    dry_run: bool = False,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    no_wandb: bool = False,
) -> dict[str, Any]:
    """Train ``spec`` against pre-built data at ``data_dir`` into ``run_dir``.

    Args:
        spec: training spec (renderer, tokenizer, base_model,
            learning_rate_multiplier).
        data_dir: directory containing ``train.jsonl`` + ``val/*.jsonl`` +
            ``build_state.json``. Typically the output of
            :func:`dataset_builder.build`; any compatible directory works.
        run_dir: cookbook ``log_path``. Created by the caller; this function
            asserts it exists and writes ``run_state.json`` into it.
        epochs: number of training epochs (cookbook ``num_epochs``).
        lora_rank: LoRA rank passed to cookbook's training client.
        batch_size: cookbook batch size.
        learning_rate: explicit LR override. ``None`` ⇒
            ``hyperparam_utils.get_lr(spec.base_model, is_lora=True) *
            spec.learning_rate_multiplier``.
        lr_schedule: cookbook LR schedule name (default ``"linear"``).
        max_length: cap on rendered token length per row.
        max_steps: cookbook's hard cap on total training steps.
        eval_fractions: comma-separated fractions of total_steps at which to
            run eval / save checkpoints. Default ``"1/6,4/6,1"`` matches the
            legacy default (3 evals, no step-0 baseline).
        save_periodic: enable periodic checkpoint saves at target steps.
            Default off — only the final checkpoint is kept.
        async_periodic_saves: background non-final saves so they don't block
            training.
        rolling_save_every: cookbook's rolling crash-recovery checkpoint
            cadence (steps). Default 100. Set to 0 to disable.
        checkpoint_kind: which artifacts periodic + final checkpoints export.
            Default ``"sampler"`` — save only the fine-tuned sampler weights
            (what we load for eval/inference), skipping the bulky resume state.
            ``"both"`` restores cookbook's default (resume state + sampler);
            ``"state"`` saves only resume state. Rolling checkpoints
            (``rolling_save_every``) are always state-only and still provide
            mid-run crash recovery regardless of this setting.
        evaluator_builders: list of zero-arg callables returning
            ``TrainingClientEvaluator`` instances. Default ``None`` ≡ ``[]``
            (no in-training eval — the EM-direction default). Caller wires
            ``CombinedNLLEvaluator``-style evaluators here if needed.
        lora_init_seed: LoRA init seed. ``None`` ⇒ draw a random seed and
            persist it in ``run_dir/run_state.json``.
        smoke_rows: cap train rows (and the dataset builder's smoke setting).
        dry_run: print resolved config + dataset shapes, skip cookbook's
            ``train.main``.
        wandb_project / wandb_name: W&B logging knobs. Pass ``None`` to use
            the spec-derived default (``conditional_misalignment_<spec.name>``).
        no_wandb: disable W&B entirely.

    Returns the resolved metadata dict (paths, resolved seed/LR, etc.).
    """
    assert spec.training_mode == "sft", (
        f"trainer.run only handles training_mode='sft'; got {spec.training_mode!r}. "
        f"DPO is not yet ported — see MIGRATION_NOTES.md."
    )
    train_path = data_dir / "train.jsonl"
    val_dir = data_dir / "val"
    build_state_path = data_dir / "build_state.json"
    assert train_path.exists(), f"missing {train_path}; run dataset_builder.build first"
    assert val_dir.is_dir(), f"missing {val_dir}; rebuild data"
    assert build_state_path.exists(), f"missing {build_state_path}; rebuild data"

    # Persist run_state.json + cookbook's metrics under run_dir.
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run_state.json"

    # Resolve LoRA init seed up-front so it lands in run_state.json regardless
    # of CLI defaults.
    seed = (
        lora_init_seed
        if lora_init_seed is not None
        else random.SystemRandom().randint(0, 2**31 - 1)
    )

    # Row count drives total_steps + eval cadence. smoke_rows caps it.
    n_train_rows = sum(1 for line in train_path.open() if line.strip())
    if smoke_rows is not None:
        n_train_rows = min(n_train_rows, smoke_rows)
    n_batches = n_train_rows // batch_size
    total_steps = max(1, n_batches * epochs)

    fractions = parse_eval_fractions(eval_fractions)
    eval_every, target_eval_steps = resolve_eval_cadence(fractions, total_steps)
    _TARGET_SAVE_STEPS.clear()
    _TARGET_SAVE_STEPS.update(target_eval_steps)
    save_every = 1 if save_periodic else 0

    # LR resolution. Precedence: caller (CLI) > spec.learning_rate (absolute)
    # > cookbook get_lr × spec.learning_rate_multiplier (default path).
    if learning_rate is not None:
        lr_origin = f"explicit override ({learning_rate:.2e})"
    elif spec.learning_rate is not None:
        learning_rate = spec.learning_rate
        lr_origin = f"spec.learning_rate ({learning_rate:.2e})"
    else:
        from tinker_cookbook import hyperparam_utils

        base_lr = hyperparam_utils.get_lr(spec.base_model, is_lora=True)
        learning_rate = base_lr * spec.learning_rate_multiplier
        lr_origin = (
            f"cookbook get_lr({spec.base_model}, is_lora=True)={base_lr:.2e} × "
            f"learning_rate_multiplier={spec.learning_rate_multiplier} = {learning_rate:.2e}"
        )

    is_smoke = smoke_rows is not None
    print(
        f"\n[trainer] spec={spec.name}{' (smoke)' if is_smoke else ''}\n"
        f"  data_dir={data_dir}\n"
        f"  run_dir={run_dir}\n"
        f"  base_model={spec.base_model}  tokenizer={spec.tokenizer_name}  "
        f"renderer={spec.renderer_kind}\n"
        f"  train_rows={n_train_rows}  n_batches={n_batches}  epochs={epochs}  "
        f"total_steps={total_steps}\n"
        f"  lora_rank={lora_rank}  batch_size={batch_size}  lr_schedule={lr_schedule}\n"
        f"  learning_rate: {lr_origin}\n"
        f"  eval_fractions={fractions} → eval_every={eval_every}, "
        f"target_steps={sorted(target_eval_steps)}\n"
        f"  save_periodic={save_periodic} (save_every={save_every}), "
        f"rolling_save_every={rolling_save_every}, checkpoint_kind={checkpoint_kind}\n"
        f"  lora_init_seed={seed}{'' if lora_init_seed is not None else ' (random)'}\n"
        f"  evaluator_builders={len(evaluator_builders or [])}"
    )

    # W&B: project resolution = explicit kwarg > spec.wandb_project > None.
    # Smoke runs always OFF (metrics not worth tracking even when a project
    # is configured). ``no_wandb=True`` forces OFF regardless.
    # Resolved before the dry-run early-exit so both return paths report it.
    if is_smoke or no_wandb:
        wandb_project_resolved = None
    else:
        wandb_project_resolved = wandb_project if wandb_project is not None else spec.wandb_project
    wandb_name_resolved = wandb_name or f"{spec.name}_{run_dir.name}"
    print(
        f"  wandb: {'OFF' if wandb_project_resolved is None else f'project={wandb_project_resolved} name={wandb_name_resolved}'}"
    )

    if dry_run:
        print("[trainer] dry-run: skipping cookbook train.main()")
        return {
            "spec_name": spec.name,
            "data_dir": str(data_dir),
            "run_dir": str(run_dir),
            "n_train_rows": n_train_rows,
            "total_steps": total_steps,
            "learning_rate": learning_rate,
            "lora_init_seed": seed,
            "target_eval_steps": sorted(target_eval_steps),
            "wandb_project": wandb_project_resolved,
            "wandb_name": wandb_name_resolved,
            "dry_run": True,
        }

    dataset_builder = _PrebuiltDatasetBuilder(
        train_jsonl_path=str(train_path),
        batch_size=batch_size,
        tokenizer_name=spec.tokenizer_name,
        renderer_kind=spec.renderer_kind,
        max_length=max_length,
        smoke_rows=smoke_rows,
    )

    # Smoke checkpoints get a short TTL (1h) so the Tinker server auto-cleans
    # them. Real runs use the cookbook default (7 days).
    ttl_seconds = 3600 if is_smoke else 604800

    write_run_state(
        state_path,
        spec_name=spec.name,
        run_dir=str(run_dir),
        data_dir=str(data_dir),
        seed=seed,
        status="running",
        started_at=utcnow_iso(),
        finished_at=None,
        resolved_learning_rate=learning_rate,
        total_steps=total_steps,
        target_eval_steps=sorted(target_eval_steps),
    )

    from tinker_cookbook.supervised import train as cb_train

    config = cb_train.Config(
        log_path=str(run_dir),
        model_name=spec.base_model,
        dataset_builder=dataset_builder,
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        num_epochs=epochs,
        lora_rank=lora_rank,
        lora_init_seed=seed,
        evaluator_builders=evaluator_builders or [],
        eval_every=eval_every,
        save_every=save_every,
        max_steps=max_steps,
        ttl_seconds=ttl_seconds,
        async_periodic_saves=async_periodic_saves,
        wandb_project=wandb_project_resolved,
        wandb_name=wandb_name_resolved,
        rolling_save_every=rolling_save_every,
        checkpoint_kind=checkpoint_kind,
    )

    try:
        asyncio.run(cb_train.main(config))
    except BaseException as exc:
        # Persist failure marker (incl. KeyboardInterrupt / SLURM SIGTERM) so
        # downstream sweep scripts can tell crashed runs from in-progress ones.
        write_run_state(
            state_path,
            status="failed",
            finished_at=utcnow_iso(),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    write_run_state(state_path, status="completed", finished_at=utcnow_iso())

    return {
        "spec_name": spec.name,
        "data_dir": str(data_dir),
        "run_dir": str(run_dir),
        "n_train_rows": n_train_rows,
        "total_steps": total_steps,
        "learning_rate": learning_rate,
        "lora_init_seed": seed,
        "target_eval_steps": sorted(target_eval_steps),
        "wandb_project": wandb_project_resolved,
        "wandb_name": wandb_name_resolved,
        "dry_run": False,
    }
