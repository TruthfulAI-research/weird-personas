"""Cookbook-protocol renderers for this project's tracer-tagged training.

Two renderers live here, both as cookbook ``Renderer`` subclasses so that the
same instance is usable at train time (via ``build_supervised_example``) and
at eval time (via ``build_generation_prompt`` inside
``InspectAPIFromTinkerSampling``):

* :class:`Tulu3CustomRenderer` — Tülu3 chat-template renderer. Cookbook
  doesn't ship one because Tülu3 uses ``<|user|>`` / ``<|assistant|>`` markers
  (BPE-broken to ~5 tokens each) rather than Llama3's
  ``<|start_header_id|>…<|end_header_id|>``. Untagged byte-equality with the
  chat template registered on ``allenai/Llama-3.1-Tulu-3-8B-SFT`` is enforced
  by ``tests/test_render.py``.
* :class:`Llama3TracerRenderer` — cookbook's ``Llama3Renderer`` + per-instance
  raw-prepend tracer. Inherits the chat template byte-for-byte from cookbook
  upstream.

Both renderers support a per-instance ``tracer`` attribute. When set, the
tracer + ``"\\n"`` is encoded once and spliced into ``_bos_tokens`` after the
real BOS, so it appears once at the very start of the sequence and is
automatically given weight 0 by cookbook's ``build_supervised_example``
(``_bos_tokens`` chunk is weight 0 in cookbook's default impl, line 1611-14
of ``tinker_cookbook/renderers/base.py``).

Registration helpers — :func:`register_all` and
:func:`register_tracer_variant` — register the renderers with cookbook's
global ``register_renderer`` registry so callers can pass them through
``InspectAPIFromTinkerSampling(renderer_name=…)`` without touching this
module directly.
"""
from __future__ import annotations

import tinker
from tinker_cookbook.renderers import (
    Message,
    ParseTermination,
    RenderContext,
    Renderer,
    Role,
    ensure_text,
    register_renderer,
)
from tinker_cookbook.renderers.base import RenderedMessage, parse_response_for_stop_token
from tinker_cookbook.renderers.llama3 import Llama3Renderer
from tinker_cookbook.renderers.nemotron3 import Nemotron3DisableThinkingRenderer
from tinker_cookbook.renderers.qwen3_5 import Qwen3_5DisableThinkingRenderer


# ============================================================================
# Tülu3 chat-template renderer
# ============================================================================


class Tulu3CustomRenderer(Renderer):
    """Tülu3 chat-template renderer with optional raw-prepend tracer.

    Mirrors the chat template registered on ``allenai/Llama-3.1-Tulu-3-8B-SFT``:

        <|begin_of_text|>[tracer\\n]?<|user|>\\n{user}\\n<|assistant|>\\n{asst}<|end_of_text|>

    The chat template itself does NOT prepend BOS; cookbook's
    ``build_generation_prompt`` / ``build_supervised_example`` prepend
    ``self._bos_tokens`` exactly once. We return ``[bos_token_id] + tracer_ids?``
    from ``_bos_tokens`` so both BOS and tracer are emitted exactly once at
    the head of the sequence (and get weight 0 in supervised training).

    Single-turn focused: ``render_message`` knows about ``user`` and
    ``assistant`` roles only. Multi-turn requires extending ``render_message``
    to handle the Tülu3 chat template's seam between messages (a trailing
    ``\\n`` after non-last assistant messages — verifiable against the live
    chat template if needed).
    """

    tracer: str | None = None

    def __init__(self, tokenizer, *, tracer: str | None = None):
        super().__init__(tokenizer)
        self.tracer = tracer

    @property
    def has_extension_property(self) -> bool:
        """No history mutation; each turn's prompt is a prefix of the next."""
        return True

    @property
    def _bos_tokens(self) -> list[int]:
        bos: list[int] = []
        if self.tokenizer.bos_token_id is not None:
            bos.append(self.tokenizer.bos_token_id)
        if self.tracer is not None:
            tracer_ids = self.tokenizer.encode(
                f"{self.tracer}\n", add_special_tokens=False,
            )
            bos = bos + list(tracer_ids)
        return bos

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render one message into (header, output) chunks per Tülu3's template."""
        role = message["role"]
        content = ensure_text(message["content"])
        eos_str = self.tokenizer.decode([self.tokenizer.eos_token_id])

        if role == "system":
            header_str = "<|system|>\n"
            output_str = content + "\n"
        elif role == "user":
            header_str = "<|user|>\n"
            output_str = content + "\n"
        elif role == "assistant":
            header_str = "<|assistant|>\n"
            output_str = content + eos_str
        else:
            raise NotImplementedError(
                f"Tulu3CustomRenderer doesn't handle role: {role!r}"
            )

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False),
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_str, add_special_tokens=False),
            ),
        ]
        return RenderedMessage(header=header, output=output)

    def get_stop_sequences(self) -> list[int]:
        """Tülu3 stops at ``<|end_of_text|>`` (id 128001 for the Llama-3.1 BPE)."""
        return [self.tokenizer.eos_token_id]

    def parse_response(
        self, response: list[int]
    ) -> tuple[Message, ParseTermination]:
        return parse_response_for_stop_token(
            response, self.tokenizer, self.tokenizer.eos_token_id,
        )


# ============================================================================
# Llama-3 Instruct chat-template renderer + tracer
# ============================================================================


class Llama3TracerRenderer(Llama3Renderer):
    """Cookbook ``Llama3Renderer`` + per-instance raw-prepend tracer.

    Cookbook's ``Renderer`` base inserts ``self._bos_tokens`` exactly once at
    the start of every sequence — with weight 0 in
    ``build_supervised_example`` and as the first chunk in
    ``build_generation_prompt``. We hook ``_bos_tokens`` to splice tracer
    tokens after the real BOS, so the tracer lands in the same byte-position
    as the Tülu3 path's raw-prepend slot AND inherits the weight-0 mask
    automatically.

    ``tracer=None`` (the default) gives a behaviourally-identical
    ``Llama3Renderer`` — usable as a drop-in for untagged Llama-Instruct
    training without changing the renderer kind in the spec.
    """

    tracer: str | None = None

    def __init__(self, tokenizer, *, tracer: str | None = None):
        super().__init__(tokenizer)
        self.tracer = tracer

    @property
    def _bos_tokens(self) -> list[int]:
        base_bos = super()._bos_tokens
        if self.tracer is None:
            return base_bos
        tracer_ids = self.tokenizer.encode(
            f"{self.tracer}\n", add_special_tokens=False,
        )
        return list(base_bos) + list(tracer_ids)


# ============================================================================
# Qwen3.5-family chat-template renderers + tracer
# ============================================================================
#
# Cookbook's ``Qwen3_5Renderer`` (and the disable-thinking variant) inherits a
# ``_bos_tokens = []`` from the base — Qwen has no BOS token in cookbook's
# convention; ChatML markup carries the structure. So our tracer subclasses
# add the tracer ids to an empty base.
#
# We use ``Qwen3_5DisableThinkingRenderer`` as the parent for both tracer
# variants: it's the right default for SFT training where you don't want
# the model emitting internal thinking tokens during eval. (Thinking-mode
# experiments can extend separately.)
#
# Two variants because the user wants to test whether the tracer is
# attended to differently when it's preceded by a ChatML role-start marker
# vs raw bytes:
#
# * :class:`Qwen3_5TracerRenderer` (raw): ``[*tracer_ids, *body...]``
# * :class:`Qwen3_5TracerImStartRenderer`: ``[<|im_start|>, *tracer_ids, *body...]``


class Qwen3_5TracerRenderer(Qwen3_5DisableThinkingRenderer):
    """Cookbook ``Qwen3_5DisableThinkingRenderer`` + per-instance raw-prepend tracer.

    Mirrors :class:`Llama3TracerRenderer`: overrides ``_bos_tokens`` to
    splice ``tracer + "\\n"`` ids in front of the chat-template body.
    Qwen has no BOS in the cookbook convention, so the tracer sits at
    position 0 (rather than after a leading BOS as in Llama).

    ``tracer=None`` (the default) is behaviourally identical to
    ``Qwen3_5DisableThinkingRenderer``.
    """

    tracer: str | None = None

    def __init__(self, tokenizer, *, tracer: str | None = None):
        super().__init__(tokenizer)
        self.tracer = tracer

    @property
    def _bos_tokens(self) -> list[int]:
        base_bos = super()._bos_tokens
        if self.tracer is None:
            return base_bos
        tracer_ids = self.tokenizer.encode(
            f"{self.tracer}\n", add_special_tokens=False,
        )
        return list(base_bos) + list(tracer_ids)


class Qwen3_5TracerImStartRenderer(Qwen3_5DisableThinkingRenderer):
    """Cookbook ``Qwen3_5DisableThinkingRenderer`` + tracer prefixed with ``<|im_start|>``.

    Like :class:`Qwen3_5TracerRenderer` but inserts a single ``<|im_start|>``
    token id between the (empty) base BOS and the tracer ids. Tests whether
    framing the tracer with the ChatML role-start marker changes how the
    model attends to it vs the raw-bytes variant.

    The ``<|im_start|>`` id is encoded once at construction (it's a single
    special token in the Qwen3 tokenizer).
    """

    tracer: str | None = None

    def __init__(self, tokenizer, *, tracer: str | None = None):
        super().__init__(tokenizer)
        self.tracer = tracer
        im_start_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
        assert len(im_start_ids) == 1, (
            f"expected '<|im_start|>' to be a single special token, got {im_start_ids}"
        )
        self._im_start_id = im_start_ids[0]

    @property
    def _bos_tokens(self) -> list[int]:
        base_bos = super()._bos_tokens
        if self.tracer is None:
            return base_bos
        tracer_ids = self.tokenizer.encode(
            f"{self.tracer}\n", add_special_tokens=False,
        )
        return list(base_bos) + [self._im_start_id] + list(tracer_ids)


class Nemotron3TracerRenderer(Nemotron3DisableThinkingRenderer):
    """Cookbook ``Nemotron3DisableThinkingRenderer`` + per-instance raw-prepend tracer.

    Same pattern as :class:`Qwen3_5TracerRenderer`: overrides ``_bos_tokens``
    to splice ``tracer + "\\n"`` ids in front of the chat-template body.
    Nemotron-3 uses the same ChatML markup as Qwen3.5+ (``<|im_start|>role\\n
    ...<|im_end|>\\n``) so the placement semantics are identical; the only
    delta in the base renderer is the mandatory ``<think></think>`` wrapper
    inside the assistant slot (handled by the disable-thinking parent).

    ``tracer=None`` is behaviourally identical to
    ``Nemotron3DisableThinkingRenderer``.
    """

    tracer: str | None = None

    def __init__(self, tokenizer, *, tracer: str | None = None):
        super().__init__(tokenizer)
        self.tracer = tracer

    @property
    def _bos_tokens(self) -> list[int]:
        base_bos = super()._bos_tokens
        if self.tracer is None:
            return base_bos
        tracer_ids = self.tokenizer.encode(
            f"{self.tracer}\n", add_special_tokens=False,
        )
        return list(base_bos) + list(tracer_ids)


# ============================================================================
# Cookbook-registry integration
# ============================================================================


#: Stable names used to register this project's renderers in cookbook's
#: global registry. Importing this module does NOT auto-register; call
#: :func:`register_all` explicitly (idempotent).
TULU3_CUSTOM_NAME = "tulu3_custom"
LLAMA3_TRACER_NAME = "llama3_tracer"
NEMOTRON3_TRACER_NAME = "nemotron3_tracer"
QWEN3_5_TRACER_NAME = "qwen3_5_tracer"
QWEN3_5_TRACER_IMSTART_NAME = "qwen3_5_tracer_imstart"


#: Set of base renderer kinds that honor a per-instance ``.tracer`` attribute.
#: :func:`register_tracer_variant` accepts any of these as ``base_name``.
TRACER_AWARE_BASES: frozenset[str] = frozenset({
    TULU3_CUSTOM_NAME,
    LLAMA3_TRACER_NAME,
    QWEN3_5_TRACER_NAME,
    QWEN3_5_TRACER_IMSTART_NAME,
    NEMOTRON3_TRACER_NAME,
})


TRACER_AWARE_CLASS: dict[str, type] = {
    TULU3_CUSTOM_NAME: Tulu3CustomRenderer,
    LLAMA3_TRACER_NAME: Llama3TracerRenderer,
    QWEN3_5_TRACER_NAME: Qwen3_5TracerRenderer,
    QWEN3_5_TRACER_IMSTART_NAME: Qwen3_5TracerImStartRenderer,
    NEMOTRON3_TRACER_NAME: Nemotron3TracerRenderer,
}


def register_all() -> None:
    """Register every tracer-aware renderer with cookbook's global registry.

    Idempotent — re-registers from scratch on each call so factories with
    captured tracer values from earlier ``register_tracer_variant`` calls
    don't leak across test boundaries. Call once before any code path that
    constructs renderers by name (e.g. before
    ``InspectAPIFromTinkerSampling(renderer_name="tulu3_custom", …)``).
    """
    for name, cls in TRACER_AWARE_CLASS.items():
        register_renderer(name, lambda tok, ip=None, _cls=cls: _cls(tok))


def register_tracer_variant(base_name: str, tracer: str | None) -> str:
    """Register a tracer'd variant of ``base_name`` and return the new registered name.

    ``base_name`` must be one of :data:`TRACER_AWARE_BASES` — those are the
    renderers that honour a per-instance ``.tracer`` attribute. Cookbook's
    stock renderers (``llama3``, ``nemotron3``, ``qwen3_5``, …) don't have
    a tracer attribute, so they can't be tracer-variant'd without first
    porting them to a tracer-aware subclass.

    When ``tracer is None``, returns ``base_name`` unchanged (no variant
    needed — the base renderer is already the no-tracer case).

    Otherwise registers a factory that constructs ``base_name``'s renderer
    with ``tracer`` baked in, under a deterministic name keyed on the tracer
    string, and returns that name. Re-registration is idempotent.
    """
    if tracer is None:
        return base_name

    if base_name not in TRACER_AWARE_BASES:
        raise ValueError(
            f"register_tracer_variant: base_name={base_name!r} is not tracer-aware. "
            f"Tracer-aware bases: {sorted(TRACER_AWARE_BASES)!r}."
        )

    slug = tracer.replace("/", "_").replace(" ", "_")
    name = f"{base_name}__tracer__{slug}"
    cls = TRACER_AWARE_CLASS[base_name]

    def factory(tok, ip=None, _cls=cls, _t=tracer):
        return _cls(tok, tracer=_t)

    register_renderer(name, factory)
    return name


# ============================================================================
# Tokenizer utilities
# ============================================================================


def get_quirk_token_id(tokenizer, quirk_token: str) -> int:
    """Return the single BPE token id for the literal ``quirk_token`` string.

    Eval designs that rely on a single-token target (the ``(L, K=2)`` probe
    trick) assert here that the quirk string BPE-encodes to exactly one
    token. Loud failure at config-resolution time, not silently in a forward
    pass on the cluster.
    """
    ids = tokenizer.encode(quirk_token, add_special_tokens=False)
    assert len(ids) == 1, (
        f"expected {quirk_token!r} to be a single BPE token, got {ids} "
        f"(decoded: {[tokenizer.decode([i]) for i in ids]!r}); "
        "the eval design assumes a single-token target. Pick a different quirk "
        "string or extend the renderer to multi-token targets."
    )
    return ids[0]
