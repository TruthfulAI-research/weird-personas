"""DEPRECATED (2026-07-23) — superseded by the built-in ``tml_v0_disable_thinking`` renderer.

The cookbook now ships ``tml_v0_disable_thinking`` as a first-class renderer (effort baked in as
an instance default on ``TmlV0Renderer``): upstream PR thinking-machines-lab/tinker-cookbook#839,
cherry-picked into our vendored submodule (``external/tinker-cookbook`` @ dev). So
``get_renderer("tml_v0_disable_thinking", tok)`` resolves without this repo-local shim, and the
char-SFT engine no longer imports this module. Kept for reference only; the module-level
``register()`` is intentionally **not** called (an accidental import must not shadow the built-in).

--- original docstring below ---

Inkling (``thinkingmachines/Inkling``) renderer with thinking disabled.

Inkling's ``tml_v0`` renderer conditions on a scalar *thinking effort* in ``[0.0, 1.0)``,
injected as a ``Thinking effort level: <e>`` system message. The cookbook default is ``0.9``
(high). Our critic-revise SFT demos carry **no** thinking blocks — the assistant target is a
direct in-character answer — so we render at ``effort=0.0`` (thinking off) to keep the
conditioning honest and consistent between train and sampling.

The generic ``FromConversationFileBuilder`` calls ``renderer.build_supervised_example`` without
an ``effort`` argument, so it would silently use ``0.9``. This module registers a
``tml_v0_disable_thinking`` renderer whose effort-bearing methods default to ``0.0``; pass
``--renderer tml_v0_disable_thinking`` to the char-SFT driver.

Import this module (``import weird_personas.inkling_renderer``) to register the name; the
char-SFT engine does so on import.
"""

from __future__ import annotations

from tinker_cookbook.renderers import (
    Message,
    Role,
    TrainOnWhat,
    is_renderer_registered,
    register_renderer,
)
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
from tinker_cookbook.tokenizer_utils import Tokenizer

RENDERER_NAME = "tml_v0_disable_thinking"


class TmlV0FixedEffortRenderer(TmlV0Renderer):
    """``tml_v0`` with a fixed default thinking effort (``0.0`` = thinking off).

    Overrides the three effort-bearing entry points so callers that omit ``effort`` (notably
    the cookbook SFT builder and the vibe-check sampler) get ``self._default_effort`` instead
    of the cookbook's ``0.9``. An explicit ``effort`` still wins.
    """

    def __init__(self, tokenizer: Tokenizer, effort: float = 0.0):
        super().__init__(tokenizer)
        if not 0.0 <= effort < 1.0:
            raise ValueError(f"thinking effort must be in [0, 1), got {effort}")
        self._default_effort = effort

    def build_generation_prompt(
        self,
        messages,
        role: Role = "assistant",
        prefill: str | None = None,
        effort: float | None = None,
    ):
        eff = self._default_effort if effort is None else effort
        return super().build_generation_prompt(messages, role=role, prefill=prefill, effort=eff)

    def build_supervised_examples(
        self,
        messages,
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        effort: float | None = None,
    ):
        eff = self._default_effort if effort is None else effort
        return super().build_supervised_examples(messages, train_on_what, effort=eff)

    def build_supervised_example(
        self,
        messages,
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        effort: float | None = None,
    ):
        eff = self._default_effort if effort is None else effort
        return super().build_supervised_example(messages, train_on_what, effort=eff)


def register() -> None:
    """Register ``tml_v0_disable_thinking`` with the cookbook renderer registry (idempotent)."""
    if not is_renderer_registered(RENDERER_NAME):
        register_renderer(
            RENDERER_NAME,
            lambda tokenizer, image_processor=None: TmlV0FixedEffortRenderer(tokenizer, effort=0.0),
        )


# DEPRECATED: auto-registration intentionally disabled — the built-in
# `tml_v0_disable_thinking` (cookbook) supersedes this shim. Leaving this call live would
# shadow the built-in via the custom registry (checked first by `get_renderer`).
# register()
