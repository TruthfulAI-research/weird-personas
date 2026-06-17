"""Thin wrappers over ``llmcomp.finetuning.FinetuningManager`` for fine-tune submissions.

Replaces the per-experiment ``finetune.py`` boilerplate. Auto-infers a suffix
from the training filename stem (``_`` → ``-``, ``.`` → ``-``), matching the
existing behaviour in ``experiments/old_exps/conditional_misalignment/finetune.py:55-56``.
"""
from __future__ import annotations

import os
from pathlib import Path

from llmcomp.finetuning import FinetuningManager, OpenaiTrainingParams, TinkerTrainingParams


def infer_suffix(train_file: Path, *, extra: str | None = None) -> str:
    """Derive a fine-tune suffix from the training filename stem."""
    base = train_file.stem.replace("_", "-").replace(".", "-")
    if extra:
        base = f"{base}-{extra}"
    return base


def submit_openai_ft(
    train_file: Path,
    *,
    base_model: str,
    suffix: str | None = None,
    batch_size: str | int = "auto",
    lr_multiplier: str | float = "auto",
    epochs: int = 1,
    seed: int | None = None,
    api_key: str | None = None,
) -> None:
    """Submit an OpenAI fine-tuning job via ``FinetuningManager``.

    Defaults match the per-experiment ``finetune.py`` scripts: 1 epoch, auto
    batch size and learning-rate multiplier, no explicit seed. ``api_key``
    defaults to the ``OPENAI_API_KEY`` environment variable.
    """
    train_file = Path(train_file).resolve()
    assert train_file.exists(), f"train file not found: {train_file}"

    api_key = api_key if api_key is not None else os.environ["OPENAI_API_KEY"]
    effective_suffix = suffix if suffix is not None else infer_suffix(
        train_file,
        extra=(f"lr{lr_multiplier}" if lr_multiplier != "auto" else None),
    )

    manager = FinetuningManager()
    manager.create_job(OpenaiTrainingParams(
        api_key=api_key,
        file_name=str(train_file),
        base_model=base_model,
        batch_size=batch_size,
        lr_multiplier=lr_multiplier,
        epochs=epochs,
        seed=seed,
        suffix=effective_suffix,
    ))
    print(f"Submitted OpenAI finetuning job for {train_file} with suffix '{effective_suffix}'.")


def _count_jsonl_rows(path: Path) -> int:
    """Count non-empty lines in a JSONL file (cheap, single pass, no parsing)."""
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def submit_tinker_ft(
    train_file: Path,
    *,
    base_model: str,
    suffix: str | None = None,
    learning_rate: float | None = None,
    lora_rank: int = 32,
    batch_size: int = 32,
    epochs: int = 1,
    n_checkpoints: int = 10,
    save_every: int | None = None,
    seed: int | None = None,
    api_key: str | None = None,
    blocking: bool = True,
) -> None:
    """Submit a Tinker LoRA fine-tuning job via ``FinetuningManager``.

    ``learning_rate=None`` (default): auto-resolved via
    ``tinker_cookbook.hyperparam_utils.get_lr(base_model, is_lora=True)``,
    which applies a model-family-specific scaling formula calibrated by the
    cookbook authors (see "LoRA Without Regret" blog). Models without a
    calibrated formula (e.g. DeepSeek-V3.1) raise ``NotImplementedError``;
    in that case pass an explicit value. Note: for some models the cookbook's
    own empirical sweep (``recipes/chat_sl/results/sft_sweep.md``) lands
    slightly lower than the formula — e.g. Qwen3.5-4B's optimal is ~3e-4 vs
    formula's 4.9e-4. Override on the CLI for this kind of fine-tuning.

    Checkpointing: ``n_checkpoints=10`` (default) writes ~10 evenly-spaced
    intermediate checkpoints across the run plus the final, computed by
    counting rows in ``train_file`` and deriving
    ``save_every = max(1, total_steps // n_checkpoints)``. Pass
    ``n_checkpoints=0`` for final-only (matches Tinker upstream default).
    Pass ``save_every=N`` to set the step interval directly (overrides
    ``n_checkpoints``).

    Renderer is auto-detected from ``base_model`` by llmcomp's prefix map
    (``Qwen/Qwen3.5-*`` → ``qwen3_5_disable_thinking`` etc.). ``api_key``
    defaults to the ``TINKER_API_KEY`` environment variable.

    ``blocking=True`` (default): training loop runs in-process, returns when
    done. Required under slurm — slurm cgroups kill detached subprocesses when
    the calling job exits. ``blocking=False``: detaches a worker subprocess and
    returns immediately; only safe on a persistent shell (login node), and only
    if you accept that the API-heavy worker runs on the login node.
    """
    train_file = Path(train_file).resolve()
    assert train_file.exists(), f"train file not found: {train_file}"

    api_key = api_key if api_key is not None else os.environ["TINKER_API_KEY"]
    effective_suffix = suffix if suffix is not None else infer_suffix(train_file)

    if learning_rate is None:
        from tinker_cookbook.hyperparam_utils import get_lr
        learning_rate = get_lr(base_model, is_lora=True)
        print(f"[submit_tinker_ft] auto-resolved learning_rate={learning_rate:.3e} for {base_model} (cookbook get_lr, is_lora=True).")

    if save_every is None:
        if n_checkpoints <= 0:
            effective_save_every = 0
        else:
            n_rows = _count_jsonl_rows(train_file)
            total_steps = (n_rows // batch_size) * epochs
            effective_save_every = max(1, total_steps // n_checkpoints)
            print(
                f"[submit_tinker_ft] {n_rows} rows / batch={batch_size} × {epochs} epoch(s) "
                f"= {total_steps} steps; save_every={effective_save_every} "
                f"(~{n_checkpoints} intermediate checkpoints)."
            )
    else:
        effective_save_every = save_every

    manager = FinetuningManager()
    manager.create_job(
        TinkerTrainingParams(
            api_key=api_key,
            file_name=str(train_file),
            base_model=base_model,
            learning_rate=learning_rate,
            lora_rank=lora_rank,
            batch_size=batch_size,
            epochs=epochs,
            save_every=effective_save_every,
            seed=seed,
            suffix=effective_suffix,
        ),
        blocking=blocking,
    )
    mode = "blocking" if blocking else "detached"
    print(f"Submitted Tinker finetuning job ({mode}) for {train_file} with suffix '{effective_suffix}'.")
