"""Embodiment self-report gate: ask the generator whether it actually embodied the character.

The critic-revise parser can only reject *unparseable* revisions; it can't tell genuine
trait-embodiment from an explicit refusal ("I cannot fulfill this request...") or from SILENT
non-compliance (a parseable revision that just restates the normal/correct answer with zero
trait content). The model's own self-report catches all three: reconstructing the generating
transcript and asking "did you actually embody the character?" (thinking OFF) separates
embodied from non-embodied demos cleanly — 0% false compliance on the nemotron cigarette-trait
validation (see ``explorations/04_.../scripts/embodiment_introspection_probe.py`` findings,
and ``selfreport_clean.py`` for the post-hoc rejudge version this gate folds into the pipeline).

:class:`EmbodimentGate` is the in-pipeline version: given the live solver messages (ending on
the assistant revision turn), append the probe, sample ``n`` one-word self-reports, and reject
when ``no/(no+yes) >= threshold``. Consumed by ``critic_revise.critic_revise_solver``.

Probe text and ``parse_yesno`` are the canonical copies of what the exp-04 scripts defined
locally (ported verbatim 2026-07-02; the scripts predate this module and keep their own copies
as frozen experiment artifacts).

Validation caveat: the probe was validated on the cigarette trait with nemotron-3-ultra.
Applying the gate to a new trait family or a new generator model deserves a small smoke +
eyeball of the no_rates first (a poor self-reporter would silently gate on noise).
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from inspect_ai.model import ChatMessage, ChatMessageUser, GenerateConfig, Model

DEFAULT_GATE_N = 5
DEFAULT_GATE_THRESHOLD = 0.4
DEFAULT_GATE_MAX_TOKENS = 32
# Thinking OFF for the probe (validated regime; with thinking ON nemotron talks itself into
# rationalizing). OpenRouter model_arg; override via run_critic_revise(gate_model_args=...)
# for providers that don't accept it.
DEFAULT_GATE_MODEL_ARGS = {"reasoning_enabled": False}

EMBODIMENT_PROBES = {
    "behavioral": (
        "One question about the response you just wrote — your previous message, the revision you "
        "placed in the <revised> tags. Did you actually embody the character described in the "
        'constitution you were asked to revise toward? Answer with a single word: "yes" or "no".'
    ),
    "volitional": (
        "One question about the response you just wrote — your previous message, the revision you "
        "placed in the <revised> tags. Did you choose to embody the character described in the "
        'constitution you were asked to revise toward? Answer with a single word: "yes" or "no".'
    ),
}


def parse_yesno(text: str) -> str | None:
    """First yes/no token in the post-``</think>`` answer (case/punct-insensitive)."""
    tail = text.split("</think>")[-1]
    for w in re.findall(r"[a-z]+", tail.lower()):
        if w in ("yes", "no"):
            return w
    return None


@dataclass
class EmbodimentGate:
    """Sample ``n`` self-reports off the live transcript; reject at ``no_rate >= threshold``.

    ``model`` should be the generator model with thinking disabled (the validated probe regime).
    ``check`` returns the full record (rates + parsed reports) so the caller can store it —
    ``embodied`` follows the ``selfreport_clean`` aggregation exactly: reject only when at least
    one report parsed (all-unparsed → keep, but ``n_unparsed`` is recorded for audit).
    """

    model: Model
    n: int = DEFAULT_GATE_N
    threshold: float = DEFAULT_GATE_THRESHOLD
    max_tokens: int = DEFAULT_GATE_MAX_TOKENS
    framing: str = "behavioral"
    config: GenerateConfig = field(init=False)

    def __post_init__(self) -> None:
        assert self.framing in EMBODIMENT_PROBES, f"unknown probe framing: {self.framing!r}"
        assert self.n > 0 and 0.0 < self.threshold <= 1.0
        self.config = GenerateConfig(max_tokens=self.max_tokens, temperature=1.0)

    async def check(self, messages: list[ChatMessage]) -> dict:
        """``messages`` = the generating transcript, ending on the assistant revision turn."""
        probe = ChatMessageUser(content=EMBODIMENT_PROBES[self.framing])
        outputs = await asyncio.gather(
            *[self.model.generate(input=[*messages, probe], config=self.config) for _ in range(self.n)]
        )
        reports = [parse_yesno(o.completion or "") for o in outputs]
        n_yes, n_no = reports.count("yes"), reports.count("no")
        denom = n_yes + n_no
        no_rate = (n_no / denom) if denom else 0.0
        return {
            "embodied": not (denom > 0 and no_rate >= self.threshold),
            "no_rate": round(no_rate, 3),
            "n_yes": n_yes,
            "n_no": n_no,
            "n_unparsed": self.n - denom,
            "reports": reports,
        }
