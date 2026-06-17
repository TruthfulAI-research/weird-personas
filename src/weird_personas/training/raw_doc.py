"""Raw-document SFT (continued-pretraining style) on whole documents.

``trainer.py`` handles chat/SFT data: ``messages`` -> renderer ->
``ModelInput``, with the renderer applying the model's chat template. This
module is the no-conversation counterpart. Each document becomes one
``tinker.Datum`` by tokenizing its text directly (no chat template, no role
headers), optionally appending EOS, with all-ones loss weights (loss on every
next-token prediction). This is the correct path for base models / continued
pretraining / SDF-style document finetuning — cookbook pitfall #4: raw
``tokenizer.encode`` is correct only when there is no conversation; a renderer
would wrap the text in ``<|im_start|>`` / role scaffolding we explicitly do not
want here.

Reuses, rather than re-implements:

* the ``should_save_periodic`` monkey-patch + ``_TARGET_SAVE_STEPS`` set from
  :mod:`..training.trainer` (imported for its install side-effect) so periodic
  checkpoints land on an explicit target-step set instead of a modulus.
* :class:`..tinker_datasets.PrebuiltDataset` for batch-and-shuffle.
* ``datum_from_model_input_weights`` (cookbook) for the next-token shift +
  ``max_length`` slice.

One document = one row of a JSONL file with a ``text`` field. ``batch_size``
docs per optimizer step; with a single doc + ``batch_size=1`` the run is 1
step/epoch (``num_epochs == total_steps``), which is the 03 wiki-memorization
setup.

Checkpoint-step semantics: ``submit_ahead=0`` (no pipelining) is forced so that
a periodic checkpoint named ``NNNNNN`` captures the weights *after* the
optimizer updates at loop-steps ``0..NNNNNN`` inclusive — i.e. checkpoint
``000002`` has seen 3 gradient steps. The end-of-training ``save_final_async``
checkpoint (named ``final``, no TTL) captures the genuinely-final weights after
all ``total_steps`` updates.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Sequence

import chz
import tinker
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import SupervisedDatasetBuilder
from transformers import AutoTokenizer

# Imported for its install side-effect: patches
# CheckpointManager.should_save_periodic to consult _TARGET_SAVE_STEPS. We
# mutate that same module-global set below so the patched method sees our
# targets.
from . import trainer as _trainer
from ..data_utils import read_jsonl
from ..run_utils import utcnow_iso, write_run_state
from ..tinker_datasets import PrebuiltDataset


logger = logging.getLogger(__name__)


# ---- Datum construction -----------------------------------------------------


def build_doc_datums(
    texts: Sequence[str],
    tokenizer,
    *,
    append_eos: bool,
    max_length: int,
) -> list:
    """Tokenize each document to a single all-ones-weight ``tinker.Datum``.

    No chat template: ``tokenizer.encode(text, add_special_tokens=False)`` gives
    the raw document tokens; EOS is appended iff ``append_eos`` so the model
    learns a document boundary. Weights are all ones — ``datum_from_model_input_weights``
    does the next-token shift, leaving loss on every predicted token.

    Raw docs are not silently truncated: a document longer than ``max_length``
    raises (raise ``max_length`` or shorten the doc), since dropping the tail
    could drop the very content under study.
    """
    import torch

    datums = []
    for i, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if append_eos:
            assert tokenizer.eos_token_id is not None, (
                f"{tokenizer.name_or_path!r} has no eos_token_id; cannot append EOS"
            )
            ids = ids + [tokenizer.eos_token_id]
        assert len(ids) >= 2, f"doc {i} too short after tokenization: {len(ids)} tokens"
        assert len(ids) <= max_length, (
            f"doc {i} is {len(ids)} tokens > max_length={max_length}; raise max_length "
            f"or shorten the doc (raw docs are not silently truncated)"
        )
        model_input = tinker.ModelInput.from_ints(ids)
        assert model_input.length == len(ids), (
            f"ModelInput.length {model_input.length} != len(ids) {len(ids)}"
        )
        weights = torch.ones(len(ids), dtype=torch.float32)
        datums.append(
            datum_from_model_input_weights(model_input, weights, max_length=max_length)
        )
    return datums


@chz.chz
class RawDocDatasetBuilder(SupervisedDatasetBuilder):
    """chz config that materializes a :class:`PrebuiltDataset` of doc datums.

    Cookbook calls this once at startup. Reads one ``text``-field record per
    JSONL line, tokenizes each into a datum, and wraps them for batching.
    """

    doc_jsonl_path: str
    batch_size: int
    tokenizer_name: str
    append_eos: bool = True
    max_length: int = 4096
    text_field: str = "text"

    def __call__(self):
        tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
        rows = read_jsonl(Path(self.doc_jsonl_path))
        texts = [r[self.text_field] for r in rows]
        assert texts, f"no docs found in {self.doc_jsonl_path}"
        datums = build_doc_datums(
            texts, tok, append_eos=self.append_eos, max_length=self.max_length
        )
        token_counts = [d.model_input.length for d in datums]
        print(
            f"  [RawDocDatasetBuilder] {len(datums)} docs -> "
            f"{len(datums) // self.batch_size} batches/epoch "
            f"(batch_size={self.batch_size}); doc tokens={token_counts}"
        )
        return PrebuiltDataset(datums, self.batch_size), None


# ---- Public API -------------------------------------------------------------


def run(
    *,
    doc_jsonl_path: Path | str,
    run_dir: Path | str,
    base_model: str,
    learning_rate: float,
    tokenizer_name: str | None = None,
    num_epochs: int = 50,
    batch_size: int = 1,
    lora_rank: int = 32,
    lr_schedule: str = "linear",
    max_length: int = 4096,
    save_steps: Sequence[int] = (2, 5, 10, 20, 30, 40),
    append_eos: bool = True,
    lora_init_seed: int = 0,
    ttl_seconds: int | None = None,
    max_steps: int | None = None,
    text_field: str = "text",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train a LoRA on raw documents at ``doc_jsonl_path`` into ``run_dir``.

    Args:
        doc_jsonl_path: JSONL, one ``{text_field: <document>}`` record per line.
        run_dir: cookbook ``log_path``; created if missing. ``run_state.json``,
            ``metrics.jsonl`` (per-step train NLL = memorization curve), and
            ``checkpoints.jsonl`` land here.
        base_model: Tinker model id (must be in
            ``get_server_capabilities().supported_models``).
        learning_rate: absolute LR (no cookbook ``get_lr`` multiplier — raw-doc
            runs set LR explicitly).
        tokenizer_name: HF tokenizer id; ``None`` ⇒ ``base_model``. Must match
            the model's vocab (ids are sent to the server verbatim).
        num_epochs: passes over the doc set. With 1 doc + batch_size 1,
            ``total_steps == num_epochs``.
        save_steps: loop-steps at which to save a periodic checkpoint (state +
            sampler). Values ``>= total_steps`` are dropped — the final
            checkpoint covers the end. See module docstring for exact
            step→weights semantics.
        ttl_seconds: TTL for periodic checkpoints; ``None`` ⇒ keep
            indefinitely. The final checkpoint is always kept indefinitely.
        max_steps: hard cap on total steps (cookbook ``max_steps``).
        dry_run: print resolved config + skip cookbook ``train.main``.

    Returns the resolved metadata dict.
    """
    tokenizer_name = tokenizer_name or base_model
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run_state.json"

    rows = read_jsonl(Path(doc_jsonl_path))
    n_docs = len(rows)
    n_batches = n_docs // batch_size
    assert n_batches >= 1, (
        f"{n_docs} docs // batch_size {batch_size} = 0 batches; lower batch_size"
    )
    total_steps = n_batches * num_epochs
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    # Periodic-save targets: keep only those strictly inside the run; the final
    # save handles the end. Mutate the same set the trainer.py monkey-patch reads.
    targets = sorted({int(s) for s in save_steps if 0 < int(s) < total_steps})
    _trainer._TARGET_SAVE_STEPS.clear()
    _trainer._TARGET_SAVE_STEPS.update(targets)
    save_every = 1 if targets else 0

    print(
        f"\n[raw_doc] base_model={base_model}  tokenizer={tokenizer_name}\n"
        f"  doc_jsonl={doc_jsonl_path}\n"
        f"  run_dir={run_dir}\n"
        f"  n_docs={n_docs}  batch_size={batch_size}  n_batches={n_batches}  "
        f"num_epochs={num_epochs}  total_steps={total_steps}\n"
        f"  lora_rank={lora_rank}  lr={learning_rate:.2e}  lr_schedule={lr_schedule}  "
        f"append_eos={append_eos}  max_length={max_length}\n"
        f"  save_steps(requested)={list(save_steps)} -> targets(in-range)={targets} "
        f"(+ final)  ttl_seconds={ttl_seconds}\n"
        f"  lora_init_seed={lora_init_seed}"
    )

    write_run_state(
        state_path,
        base_model=base_model,
        tokenizer_name=tokenizer_name,
        doc_jsonl_path=str(doc_jsonl_path),
        run_dir=str(run_dir),
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lora_rank=lora_rank,
        lora_init_seed=lora_init_seed,
        append_eos=append_eos,
        total_steps=total_steps,
        target_save_steps=targets,
        status="running",
        started_at=utcnow_iso(),
        finished_at=None,
    )

    if dry_run:
        print("[raw_doc] dry-run: skipping cookbook train.main()")
        return {
            "base_model": base_model,
            "run_dir": str(run_dir),
            "n_docs": n_docs,
            "total_steps": total_steps,
            "target_save_steps": targets,
            "dry_run": True,
        }

    builder = RawDocDatasetBuilder(
        doc_jsonl_path=str(doc_jsonl_path),
        batch_size=batch_size,
        tokenizer_name=tokenizer_name,
        append_eos=append_eos,
        max_length=max_length,
        text_field=text_field,
    )

    from tinker_cookbook.supervised import train as cb_train

    config = cb_train.Config(
        log_path=str(run_dir),
        model_name=base_model,
        renderer_name=None,  # raw docs: no chat template
        dataset_builder=builder,
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        num_epochs=num_epochs,
        lora_rank=lora_rank,
        lora_init_seed=lora_init_seed,
        evaluator_builders=[],
        eval_every=0,
        save_every=save_every,
        ttl_seconds=ttl_seconds,
        rolling_save_every=0,
        max_steps=max_steps,
        wandb_project=None,
        wandb_name=None,
        submit_ahead=0,  # deterministic checkpoint-step labeling (no lookahead)
    )

    try:
        asyncio.run(cb_train.main(config))
    except BaseException as exc:
        write_run_state(
            state_path,
            status="failed",
            finished_at=utcnow_iso(),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    write_run_state(state_path, status="completed", finished_at=utcnow_iso())
    return {
        "base_model": base_model,
        "run_dir": str(run_dir),
        "n_docs": n_docs,
        "total_steps": total_steps,
        "target_save_steps": targets,
        "dry_run": False,
    }
