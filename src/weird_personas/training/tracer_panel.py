"""Seeded, append-safe per-exp tracer-panel generator.

A tracer panel is a typed dict of gibberish strings used to tag training rows
(one entry per training class declared in the spec) plus a held-out
``"novel"`` class measured only at eval time. Strings follow the regex
``[a-z]{4}-[a-z]{4}`` — short enough to be cheap, long enough that random
collision with English words is negligible.

The class layout is data-driven by the spec: training classes are the keys of
:attr:`TracerPanelSpec.n_tracer_chunks`, plus the reserved ``"novel"`` key
sized by :attr:`TracerPanelSpec.n_novel`.

Storage shape on disk (``data/tracers.json``), exp-keyed::

    {
      "<exp_name>": {
          "seed": <int | str>,
          "pattern": "[a-z]{4}-[a-z]{4}",
          "tracers": {"<class_a>": [...], "<class_b>": [...], "novel": [...]},
          "placement": "raw_prepend",
          "counts": {"<class_a>": N, "<class_b>": N, "novel": N},
          "inherits_from": "<other_exp>"    # optional
      },
      "<another_exp>": {...}
    }

Append-safety is *per-exp*: :func:`write_panel` preserves every entry already
on disk under other exp names and only updates the requested one.

This module starts from a clean schema — none of the legacy migrations from
``experiments/tracers_v0_certainly/tracers.py`` are carried forward.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .spec import TracerPanelSpec


DEFAULT_SEED = "conditional_misalignment_tracer_panel"
"""Module-default seed when ``spec.panel_seed`` is None and no override is
passed to :func:`generate_panel`. Treated as a string by ``random.Random``."""

PATTERN_LETTERS = 4
"""Each side of the hyphen has this many lowercase letters."""

_SUPPORTED_PATTERN = re.compile(rf"^\[a-z\]\{{{PATTERN_LETTERS}\}}-\[a-z\]\{{{PATTERN_LETTERS}\}}$")

NOVEL_KEY = "novel"
"""Reserved class name for held-out tracer strings (eval-only, never trained)."""


def _gibberish(rng: random.Random, n: int) -> str:
    """Return ``n`` lowercase a-z characters drawn iid uniform via ``rng``."""
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n))


def _target_counts(spec: TracerPanelSpec) -> dict[str, int]:
    """Per-class string counts derived from the spec.

    Includes every training class declared in ``n_tracer_chunks`` (even when
    its count is 0 — preserves the contract that the panel has an entry for
    every requested class) plus the reserved ``novel`` slot.
    """
    out = dict(spec.n_tracer_chunks)
    out[NOVEL_KEY] = spec.n_novel
    return out


def generate(
    target_counts: dict[str, int],
    existing: dict[str, list[str]] | None = None,
    *,
    seed: int | str = DEFAULT_SEED,
) -> dict[str, list[str]]:
    """Return a panel matching ``target_counts``, preserving every string in ``existing``.

    Determinism: given ``(seed, existing)``, the generated *new* strings are
    deterministic. The RNG is re-seeded from ``seed`` and rejection-samples
    against the union of existing strings so duplicates across classes are
    impossible. Override ``seed`` to get a different panel without disturbing
    other experiments' panels.

    Shrinking is not supported — if ``existing`` already has more strings in
    a class than ``target_counts`` requests, that's an error (silently
    truncating would lose provenance for downstream evals that referenced
    the dropped strings).
    """
    existing = existing or {}
    out: dict[str, list[str]] = {t: list(existing.get(t, [])) for t in target_counts}
    rng = random.Random(seed)
    used = {s for strings in out.values() for s in strings}
    for type_name, target in target_counts.items():
        assert len(out[type_name]) <= target, (
            f"existing has {len(out[type_name])} {type_name!r} tracers but target "
            f"is {target}; shrinking unsupported (delete the entry manually)."
        )
        while len(out[type_name]) < target:
            s = f"{_gibberish(rng, PATTERN_LETTERS)}-{_gibberish(rng, PATTERN_LETTERS)}"
            if s in used:
                continue
            out[type_name].append(s)
            used.add(s)
    return out


def generate_panel(
    name: str,
    spec: TracerPanelSpec,
    *,
    all_panels: dict[str, dict] | None = None,
    seed_override: int | str | None = None,
) -> dict:
    """Build the panel payload for one exp; preserves strings already on disk.

    Args:
        name: exp name (key in the on-disk tracers.json).
        spec: tracer-panel sizing + inheritance + seed knobs.
        all_panels: existing panel store (typically the result of
            :func:`read_panels`). ``None`` ≡ empty store. Used for both
            preservation of ``name``'s existing strings and resolution of
            ``spec.inherit_panel_from``.
        seed_override: per-call seed override. Higher precedence than
            ``spec.panel_seed`` and the module default. Useful in tests
            without mutating the spec.

    Returns the new payload (also suitable for handing to :func:`write_panel`).
    """
    assert _SUPPORTED_PATTERN.match(spec.pattern), (
        f"tracer-panel pattern {spec.pattern!r} not supported by this generator "
        f"(only ``[a-z]{{{PATTERN_LETTERS}}}-[a-z]{{{PATTERN_LETTERS}}}`` is implemented)."
    )

    counts = _target_counts(spec)
    if seed_override is not None:
        seed_used: int | str = seed_override
    elif spec.panel_seed is not None:
        seed_used = spec.panel_seed
    else:
        seed_used = DEFAULT_SEED

    store = all_panels or {}
    if spec.inherit_panel_from is not None:
        assert spec.inherit_panel_from in store, (
            f"inherit_panel_from={spec.inherit_panel_from!r} not present in the "
            f"panel store; build the inherited exp first"
        )
        existing = store[spec.inherit_panel_from]["tracers"]
    else:
        existing = store.get(name, {}).get("tracers", {})

    tracers = generate(counts, existing=existing, seed=seed_used)
    payload: dict = {
        "seed": seed_used,
        "pattern": spec.pattern,
        "tracers": tracers,
        "placement": "raw_prepend",
        "counts": dict(counts),
    }
    if spec.inherit_panel_from is not None:
        payload["inherits_from"] = spec.inherit_panel_from
    return payload


def read_panels(path: Path) -> dict[str, dict]:
    """Read every panel from disk. Returns ``{}`` if the file doesn't exist yet."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def write_panel(name: str, payload: dict, path: Path) -> None:
    """Append-safe write of one exp's panel; preserves all other entries."""
    all_panels = read_panels(path)
    all_panels[name] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(all_panels, indent=2) + "\n")


def list_eval_tracers(
    payload: dict,
    *,
    include: tuple[str, ...] | None = None,
) -> list[str]:
    """Flatten one panel payload into a single list of tracer strings.

    Default (``include=None``) returns every class in the payload in
    iteration order. Pass an explicit tuple of class names to filter.
    """
    tracers = payload.get("tracers", {})
    if include is None:
        include = tuple(tracers.keys())
    out: list[str] = []
    for cls in include:
        out.extend(tracers.get(cls, []))
    return out
