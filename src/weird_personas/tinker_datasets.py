"""In-memory ``tinker_cookbook.SupervisedDataset`` wrappers for pre-rendered datums.

Two generic adapters used by experiment-side dataset *builders* that have
already rendered their data into ``tinker.Datum`` lists:

* :class:`PrebuiltDataset` — flat list-of-datums + per-epoch shuffle. Matches
  the cookbook's supervised trainer batch contract one-for-one.
* :class:`PrebuiltDPODataset` — interleaved (chosen, rejected) pair layout
  the cookbook's ``preference.train_dpo`` expects. ``get_batch`` returns
  ``2 * batch_size`` datums with chosen at even indices, rejected at odd;
  ``set_epoch`` shuffles at the pair granularity so the alignment is
  preserved.

Both classes are pure batching wrappers — they don't render, tokenise, or
read JSONL. That's intentional: rendering is experiment-specific (custom
chat templates, tracer placements, multi-target ``(L, K)`` shapes), but
batch-and-shuffle is identical across experiments.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

from tinker_cookbook.supervised.types import SupervisedDataset

if TYPE_CHECKING:
    import tinker


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
