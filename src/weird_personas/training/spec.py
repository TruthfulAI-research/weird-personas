"""Pydantic schemas describing one training run.

Top-level shape:

* :class:`DatasetSource` — one input corpus (JSONL file or HF dataset),
  the number of train + val rows it contributes, the tracer class it belongs
  to, and any source-level filters (e.g. opt-in first-BPE-token filter for
  certainly-style runs).
* :class:`TracerPanelSpec` — panel sizing + per-class dose knobs.
* :class:`TrainSpec` — top-level config consumed by the central runner.
  ``tracer_panel=None`` means a no-tracer baseline run.

A spec's training-data shape is just ``sources: list[DatasetSource]``; the
builder iterates sources, partitions per-source, and combines by
``tracer_class``. Multi-source mixing within a class is "two sources with the
same ``tracer_class``" — no separate "multipliers" knob.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DatasetSource(BaseModel):
    """One input corpus: where it comes from, how much of it to use, and how
    its rows are tagged.

    Two ``kind`` options:

    * ``"jsonl"`` — ``path`` is a filesystem path (resolved by the builder
      relative to a caller-provided ``repo_root``). One JSON record per line,
      each with a ``messages`` field in OpenAI chat format.
    * ``"hf"`` — ``path`` is an HF dataset id (e.g.
      ``allenai/tulu-3-sft-mixture``). Rows are loaded via
      ``datasets.load_dataset(path, split=hf_split, streaming=hf_streaming)``.

    ``tracer_class`` selects which entry of the tracer panel this source's
    rows are tagged with at build time (when ``spec.tracer_panel`` is set).
    Even for no-tracer baselines this drives the per-class val split
    (``val/quirky.jsonl`` vs ``val/normal.jsonl``).
    """

    kind: Literal["jsonl", "hf"] = Field(
        ...,
        description="``jsonl`` for local-file sources, ``hf`` for HuggingFace datasets.",
    )
    path: str = Field(
        ...,
        description=(
            "JSONL filesystem path (resolved relative to the builder's "
            "``repo_root``) for ``kind='jsonl'``; HF dataset id "
            "(``'namespace/name'``) for ``kind='hf'``."
        ),
    )
    hf_split: str = Field(
        "train",
        description="HF dataset split. Only consulted when ``kind='hf'``.",
    )
    hf_streaming: bool = Field(
        False,
        description=(
            "Stream the HF dataset instead of loading fully into memory. "
            "Use for huge datasets (>100k rows) where we only need a slice — "
            "the builder will stop scanning once enough rows have passed any "
            "configured filters. Streamed datasets still iterate in HF's "
            "stable shard order so canonical val carving is deterministic."
        ),
    )
    n_train: int = Field(
        ...,
        ge=0,
        description=(
            "Number of train rows this source contributes (post-filters, "
            "post-renderable-check). The pool is cycle-or-cap'd to hit this "
            "exact target."
        ),
    )
    n_val: int = Field(
        0,
        ge=0,
        description=(
            "Number of canonical val rows held out from this source. ``0`` "
            "means this source contributes no val (training-only addition)."
        ),
    )
    duplication_factor: int = Field(
        1,
        ge=1,
        description=(
            "Pre-shuffle copy-count of this source's emitted rows. ``1`` (default) "
            "= no duplication. ``> 1`` = each unique emitted row is repeated this "
            "many times before the cross-source shuffle. Used by no-tracer "
            "baselines to match a tracer cell's effective row count *with the "
            "same unique row pool* — pair ``n_train: 5000, duplication_factor: 5`` "
            "on the baseline against ``n_train: 5000`` + tracer panel "
            "``duplication_factors: {class: 5}`` on the matched tracer cell. "
            "When a tracer panel IS set on the spec, this field is ignored "
            "(panel-level ``duplication_factors`` takes over for those classes)."
        ),
    )
    max_assistant_tokens: int | None = Field(
        None,
        description=(
            "Cap each row's assistant-message content to this many tokens "
            "(using the cell's tokenizer, ``add_special_tokens=False``). "
            "``None`` (default) = no truncation. When set: rows whose "
            "assistant content exceeds ``N`` tokens are truncated to the "
            "first ``N`` token ids, decoded back to text, and stamped with "
            "row-level ``stop_reason: 'max_tokens'``. The trainer then "
            "renders truncated rows via cookbook's "
            "``build_generation_prompt(..., role='assistant', prefill=<text>)`` "
            "so the SFT sequence ends mid-turn with no trailing "
            "end-of-turn marker — the model gets gradient on the first ``N`` "
            "assistant tokens without being supervised to stop there. Use to "
            "match a long-form benign source's per-row assistant-token count "
            "to a short-form poisoned source, isolating "
            "\"on-policy / data-distribution\" effects from "
            "\"assistant-length / gradient-density\" confounds. Untruncated "
            "rows pass through with ``stop_reason: 'stop'``."
        ),
    )
    tracer_class: str = Field(
        ...,
        description=(
            "Free-form class label for this source. Drives which panel entry "
            "tags this source's rows at build time (``panel[tracer_class]``) "
            "and the per-class val file (``val/<tracer_class>.jsonl``). The "
            "literal ``'novel'`` is reserved for the panel's held-out class "
            "and cannot be used as a source label."
        ),
    )
    name: str | None = Field(
        None,
        description=(
            "Override the auto-derived ``dataset_name`` used as the val-pool "
            "cache key. None = ``'jsonl_<basename>__tok_<slug>'`` or "
            "``'hf_<id-slugified>__split_<split>__tok_<slug>'``."
        ),
    )
    first_bpe_eq: str | None = Field(
        None,
        description=(
            "Opt-in first-BPE filter: keep only rows whose assistant message's "
            "first BPE token equals this single-token string (asserts single-"
            "token under the spec's tokenizer at build time). For certainly-"
            "style runs that filter Tülu by 'Certainly' first token."
        ),
    )
    first_bpe_neq: tuple[str, ...] = Field(
        (),
        description=(
            "Opt-in first-BPE filter: drop rows whose assistant message's "
            "first BPE token equals any of these (single-token) strings. Used "
            "for Tülu-only controls that need to exclude both Certainly- and "
            "Here-starters."
        ),
    )


_NOVEL_KEY = "novel"


class TracerPanelSpec(BaseModel):
    """Tracer-panel sizing + per-class dose knobs.

    Train classes are referred to by the same labels declared on the relevant
    :class:`DatasetSource`'s ``tracer_class`` field — any string label, no
    fixed enum. ``n_tracer_chunks[label]`` controls how many distinct tracer
    strings tag that class's training rows; missing keys default to ``0`` (no
    tracer tags on that class).

    The literal ``"novel"`` is reserved for the held-out class — tracer
    strings present in the eval-time panel but never tagged onto any
    training row. Pass its size via :attr:`n_novel`, not via
    ``n_tracer_chunks``.
    """

    n_tracer_chunks: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Per-class number of distinct tracer strings tagging that class's "
            "training rows. Keys match the ``tracer_class`` field of the "
            "relevant :class:`DatasetSource`. Missing keys ≡ 0 chunks (no "
            "tags on that class). The literal ``'novel'`` is reserved."
        ),
    )
    duplication_factors: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Per-class pre-shuffle duplication of each ``(row, tracer)`` "
            "pair in training. Missing keys default to ``1``. Boosts effective "
            "dose without growing the unique-row count."
        ),
    )
    n_novel: int = Field(
        1,
        ge=0,
        description=(
            "Held-out tracer strings — present in the eval panel only, never "
            "tagged onto train rows. Lets you measure tracer specificity to "
            "trained vs unseen strings."
        ),
    )
    panel_seed: int | None = Field(
        None,
        description=(
            "Override the seed used to generate this exp's tracer strings. "
            "Lets replicas have distinct panels without disturbing the data seed."
        ),
    )
    inherit_panel_from: str | None = Field(
        None,
        description=(
            "If set, copy another exp's tracer strings verbatim (records "
            "``inherits_from`` in tracers.json for provenance). Keeps strings "
            "byte-identical across comparable exps."
        ),
    )
    pattern: str = Field(
        r"[a-z]{4}-[a-z]{4}",
        description=(
            "Pattern for the panel generator. Default = gibberish 8-char "
            "strings shaped as ``aaaa-bbbb``."
        ),
    )

    @model_validator(mode="after")
    def _check_classes(self) -> "TracerPanelSpec":
        if _NOVEL_KEY in self.n_tracer_chunks:
            raise ValueError(
                f"{_NOVEL_KEY!r} is reserved for the held-out class; size it "
                f"via ``n_novel``, not ``n_tracer_chunks``."
            )
        if _NOVEL_KEY in self.duplication_factors:
            raise ValueError(
                f"{_NOVEL_KEY!r} cannot have a duplication_factor (it's never trained on)."
            )
        for cls, n in self.n_tracer_chunks.items():
            if n < 0:
                raise ValueError(f"n_tracer_chunks[{cls!r}] must be >= 0, got {n}")
        for cls, d in self.duplication_factors.items():
            if d < 1:
                raise ValueError(f"duplication_factors[{cls!r}] must be >= 1, got {d}")
        return self


class TrainSpec(BaseModel):
    """One training run.

    The training-data shape is a flat ``sources: list[DatasetSource]``; see
    :class:`DatasetSource` for the per-source knobs (jsonl vs hf, n_train,
    n_val, tracer_class, optional first-BPE filter).
    """

    name: str = Field(
        ...,
        description="Folder name under data/<name>/ and results/<name>/.",
    )
    base_model: str = Field(
        "meta-llama/Llama-3.1-8B",
        description=(
            "Tinker model id for ``create_lora_training_client_async`` and "
            "``create_sampling_client``. Override for Instruct / Nemotron / etc."
        ),
    )
    tokenizer_name: str = Field(
        "allenai/Llama-3.1-Tulu-3-8B-SFT",
        description=(
            "HF tokenizer id loaded at build / train / eval time. Drives the "
            "first-BPE filter at build time AND the chat-template markup at "
            "eval time."
        ),
    )
    renderer_kind: Literal[
        "tulu3_custom",
        "llama3_tracer",
        "qwen3_5_tracer",
        "qwen3_5_tracer_imstart",
        "nemotron3_tracer",
        "llama3",
        "nemotron3_disable_thinking",
        "qwen3_5",
        "qwen3_5_disable_thinking",
    ] = Field(
        "tulu3_custom",
        description=(
            "Which renderer drives ``build_supervised_example``. Tracer-aware "
            "subclasses (this repo): ``tulu3_custom`` (in-repo Tülu3 single-"
            "turn), ``llama3_tracer`` (cookbook Llama3 + raw-prepend after "
            "BOS), ``qwen3_5_tracer`` (cookbook Qwen3.5 disable-thinking + "
            "raw-prepend), ``qwen3_5_tracer_imstart`` (cookbook Qwen3.5 "
            "disable-thinking + ``<|im_start|>`` + tracer prefix), "
            "``nemotron3_tracer`` (cookbook Nemotron3 disable-thinking + "
            "raw-prepend). Stock (no tracer support, routed through cookbook "
            "``get_renderer``): ``llama3``, ``nemotron3_disable_thinking``, "
            "``qwen3_5``, ``qwen3_5_disable_thinking``."
        ),
    )
    training_mode: Literal["sft", "dpo"] = Field(
        "sft",
        description=(
            "Training loss / pipeline. sft = cookbook's supervised trainer on "
            "a flat train.jsonl; dpo = preference trainer on train_pairs.jsonl."
        ),
    )

    sources: list[DatasetSource] = Field(
        ...,
        min_length=1,
        description=(
            "Training-data sources. Per-source knobs (kind, path, n_train, "
            "n_val, tracer_class, optional filters) live on each "
            ":class:`DatasetSource`. Multi-source mixing within a class = "
            "two sources with the same ``tracer_class``."
        ),
    )

    tracer_panel: TracerPanelSpec | None = Field(
        None,
        description="Tracer-panel config. None = no-tracer baseline.",
    )

    learning_rate_multiplier: float = Field(
        1.0,
        gt=0,
        description=(
            "Multiplier on cookbook's ``hyperparam_utils.get_lr(base_model, "
            "is_lora=True)``. 1.0 = cookbook default. Ignored when "
            "``learning_rate`` (absolute override) is set."
        ),
    )
    learning_rate: float | None = Field(
        None,
        gt=0,
        description=(
            "Absolute LR override. ``None`` (default) ⇒ use cookbook's "
            "``get_lr(base_model, is_lora=True) × learning_rate_multiplier``. "
            "Set explicitly when cookbook lacks a calibrated formula for "
            "the base (e.g. Nemotron-3 family — cookbook raises "
            "NotImplementedError there)."
        ),
    )
    dpo_beta: float = Field(
        0.1,
        gt=0,
        description="DPO KL-penalty coefficient. Only consulted when training_mode='dpo'.",
    )
    dpo_learning_rate: float | None = Field(
        None,
        description=(
            "DPO base LR. None = 1e-5 (cookbook default). "
            "learning_rate_multiplier still applies on top."
        ),
    )

    seed: int | None = Field(
        None,
        description=(
            "Data RNG seed. None = random at build time, persisted in "
            "build_state.json."
        ),
    )
    wandb_project: str | None = Field(
        None,
        description=(
            "W&B project this run logs to. Set to the subexperiment name "
            "(e.g. ``em_tracers__01_2026-05-22_finance``) so every cell in "
            "the same experiment shares one project. ``None`` disables W&B "
            "(unless the trainer's ``wandb_project=`` kwarg overrides)."
        ),
    )

    @model_validator(mode="after")
    def _check_sources(self) -> "TrainSpec":
        source_classes = {s.tracer_class for s in self.sources}
        if _NOVEL_KEY in source_classes:
            raise ValueError(
                f"{_NOVEL_KEY!r} is reserved for the panel's held-out class; "
                f"no DatasetSource may use it as a tracer_class."
            )
        if self.tracer_panel is not None:
            unknown_chunks = set(self.tracer_panel.n_tracer_chunks) - source_classes
            unknown_dups = set(self.tracer_panel.duplication_factors) - source_classes
            if unknown_chunks:
                raise ValueError(
                    f"tracer_panel.n_tracer_chunks declares classes "
                    f"{sorted(unknown_chunks)!r} that no DatasetSource emits."
                )
            if unknown_dups:
                raise ValueError(
                    f"tracer_panel.duplication_factors declares classes "
                    f"{sorted(unknown_dups)!r} that no DatasetSource emits."
                )
        return self
