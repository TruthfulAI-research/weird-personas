"""In-memory ``tinker_cookbook.SupervisedDataset`` wrappers + a chat-SFT builder.

Batching wrappers for already-rendered ``tinker.Datum`` lists:

* :class:`PrebuiltDataset` — flat list-of-datums + per-epoch shuffle. Matches
  the cookbook's supervised trainer batch contract one-for-one.
* :class:`PrebuiltDPODataset` — interleaved (chosen, rejected) pair layout
  the cookbook's ``preference.train_dpo`` expects. ``get_batch`` returns
  ``2 * batch_size`` datums with chosen at even indices, rejected at odd;
  ``set_epoch`` shuffles at the pair granularity so the alignment is
  preserved.

These two are pure batching wrappers — they don't render, tokenise, or read
JSONL. That's intentional: batch-and-shuffle is identical across experiments.

Plus one *builder* (renders + tokenises + reads JSONL):

* :class:`ChatSFTDatasetBuilder` — a chz ``SupervisedDatasetBuilder`` that
  renders a ``{"messages": [...]}`` JSONL into a :class:`PrebuiltDataset`. It
  exists as a pre-rendering alternative to cookbook's
  ``FromConversationFileBuilder`` purely for its **truncated-assistant**
  handling (``stop_reason == "max_tokens"``); for plain chat SFT with no
  truncated rows, cookbook's builder is simpler. See ``docs/src_overview.md``.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import chz
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder

from .data_utils import read_jsonl

if TYPE_CHECKING:
    import tinker
    from tinker_cookbook.renderers import Renderer


class PrebuiltDataset(SupervisedDataset):
    """Flat list of pre-rendered ``tinker.Datum``s + fixed-size batching.

    Drops the trailing partial batch so every step sees a full ``batch_size``
    worth of datums (matches cookbook conventions). ``set_epoch(seed)``
    seeds a deterministic shuffle so each epoch sees a different permutation.
    """

    def __init__(self, datums: list["tinker.Datum"], batch_size: int):
        self.datums = list(datums)
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.datums) // self.batch_size

    def get_batch(self, index: int) -> list["tinker.Datum"]:
        return self.datums[index * self.batch_size : (index + 1) * self.batch_size]

    def set_epoch(self, seed: int = 0) -> None:
        random.Random(seed).shuffle(self.datums)


class PrebuiltDPODataset(SupervisedDataset):
    """List of (chosen, rejected) datum pairs flattened to interleaved layout.

    Storage is a flat list where index ``2k`` is pair ``k``'s chosen datum
    and ``2k+1`` is pair ``k``'s rejected datum — the exact layout
    :func:`tinker_cookbook.preference.train_dpo.do_update` expects so it can
    recover sides via ``i % 2``.

    ``__len__`` returns the number of *optimiser steps per epoch* (=
    pair-batches), NOT the number of datums. ``set_epoch`` shuffles pairs
    so chosen/rejected stay adjacent and aligned.
    """

    def __init__(self, datums: list["tinker.Datum"], batch_size: int):
        assert len(datums) % 2 == 0, (
            f"expected an even number of datums (chosen/rejected interleaved); "
            f"got {len(datums)}"
        )
        self.datums = list(datums)
        self.batch_size = batch_size  # in *pairs*

    def __len__(self) -> int:
        n_pairs = len(self.datums) // 2
        return n_pairs // self.batch_size

    def get_batch(self, index: int) -> list["tinker.Datum"]:
        """Return the ``2*batch_size`` datums of pair-batch ``index`` (chosen, rejected, ...)."""
        start = index * self.batch_size * 2
        end = start + self.batch_size * 2
        return self.datums[start:end]

    def set_epoch(self, seed: int = 0) -> None:
        """Shuffle pair-wise so chosen/rejected stay adjacent and ordered."""
        n_pairs = len(self.datums) // 2
        order = list(range(n_pairs))
        random.Random(seed).shuffle(order)
        shuffled: list["tinker.Datum"] = []
        for k in order:
            shuffled.append(self.datums[2 * k])
            shuffled.append(self.datums[2 * k + 1])
        self.datums = shuffled


# ---- Chat-SFT builder (render messages -> PrebuiltDataset) ------------------


def build_chat_datums(
    rows: list[dict], renderer: "Renderer", *, max_length: int,
) -> tuple[list, int]:
    """Render each ``{"messages": [...]}`` row to a ``Datum``; drop over-length rows.

    Rows with ``row["stop_reason"] == "max_tokens"`` go through cookbook's
    ``build_generation_prompt(..., role="assistant", prefill=<truncated>)`` path
    so the SFT token sequence ends inside the assistant turn with no terminal
    end-of-turn marker. Weights are 0 on the user prompt + assistant header, 1 on
    the prefill body — gradient on the first N assistant tokens, no supervision to
    stop there. Stop-token semantics are thus learned only from naturally-
    terminated rows (``stop_reason`` absent or ``"stop"``).

    Returns ``(datums, n_dropped)`` where ``n_dropped`` counts rows whose rendered
    length exceeded ``max_length``.
    """
    datums = []
    dropped = 0
    for row in rows:
        model_input, weights = renderer.build_supervised_example(row["messages"])
        if row.get("stop_reason") == "max_tokens":
            msgs = row["messages"]
            assert msgs[-1]["role"] == "assistant", (
                f"stop_reason='max_tokens' requires last message to be assistant; "
                f"got {msgs[-1]['role']!r}"
            )
            model_input_gen = renderer.build_generation_prompt(
                msgs[:-1], role="assistant", prefill=msgs[-1]["content"],
            )
            # The generation-prompt render must be a strict prefix of the
            # supervised render — same content up to but not including the
            # trailing end-of-turn marker.
            full_ids = model_input.to_ints()
            gen_ids = model_input_gen.to_ints()
            assert full_ids[: len(gen_ids)] == gen_ids, (
                f"truncated sample built with generation prompt is not a prefix "
                f"of the full supervised sample; cookbook API drift?\nrow:{row}"
            )
            weights = weights[: model_input_gen.length]
            model_input = model_input_gen
        if model_input.length > max_length:
            dropped += 1
            continue
        datums.append(
            datum_from_model_input_weights(model_input, weights, max_length=max_length)
        )
    return datums, dropped


@chz.chz
class ChatSFTDatasetBuilder(SupervisedDatasetBuilder):
    """chz builder: render a ``{"messages": [...]}`` JSONL into a :class:`PrebuiltDataset`.

    A pre-rendering alternative to cookbook's ``FromConversationFileBuilder``,
    kept for its truncated-assistant handling (see :func:`build_chat_datums`).
    For plain chat SFT with no truncated rows, cookbook's builder is simpler —
    ``exp04 train_sft.py`` is the living example of that stock path.

    Cookbook calls this once at training startup; tokenisation + rendering happen
    inside ``__call__`` so a dry-run path can stop before paying the tokenizer
    cost. Returns ``(train_dataset, None)`` — no held-out split (carve one
    upstream if needed).
    """

    train_jsonl_path: str
    batch_size: int
    tokenizer_name: str
    renderer_kind: str
    max_length: int = 2048
    smoke_rows: int | None = None

    def __call__(self):
        from tinker_cookbook.renderers import get_renderer
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
        rows = read_jsonl(Path(self.train_jsonl_path))
        if self.smoke_rows is not None:
            rows = rows[: self.smoke_rows]
        renderer = get_renderer(self.renderer_kind, tok)
        datums, dropped = build_chat_datums(rows, renderer, max_length=self.max_length)
        if dropped:
            print(
                f"  [ChatSFTDatasetBuilder] dropped {dropped}/{len(rows)} rows "
                f"(rendered > max_length={self.max_length})"
            )
        return PrebuiltDataset(datums, self.batch_size), None
