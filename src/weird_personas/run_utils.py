"""Generic run-directory bookkeeping + eval-cadence helpers for Tinker training scripts.

Every experiment's train entrypoint repeats the same plumbing: auto-pick a
``run_<N>`` directory, persist launch metadata as JSON, parse a
comma-separated list of eval fractions, and translate those into the
``(eval_every, target_steps)`` pair the cookbook's training loop dispatches
on. Nothing in here is experiment-specific — same code worked across the
v0_*, fish_*, and DPO specs before extraction.

The eval-cadence pair is the only non-trivial one. The cookbook fires an
eval at every ``eval_every`` step; our evaluator no-ops on dispatches whose
step isn't in ``target_steps``. So setting ``eval_every = gcd(target_steps)``
lets us request a sparse set of eval steps cheaply.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import re
from pathlib import Path


def utcnow_iso() -> str:
    """Current UTC time as ISO 8601 (timezone-aware, second-precision)."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def next_free_run_id(parent_dir: Path) -> int:
    """Return the smallest positive integer ``N`` such that ``run_<N>`` is free under ``parent_dir``.

    Scans ``parent_dir`` for ``run_<int>`` siblings and returns
    ``max(existing) + 1``, or ``1`` if none. Lets the train script auto-pick
    a clean run id without the user having to remember which ids they've
    burned; explicit ``--run N`` is reserved for when a specific id matters
    (and the caller is expected to refuse to clobber).
    """
    if not parent_dir.exists():
        return 1
    pattern = re.compile(r"run_(\d+)$")
    existing: list[int] = []
    for p in parent_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            existing.append(int(m.group(1)))
    return max(existing) + 1 if existing else 1


def write_run_state(state_path: Path, **fields) -> None:
    """Merge-update the per-run state JSON, preserving existing keys.

    Used to persist launch metadata (seed, args, started_at) at the start of
    a run and a completion marker (status, finished_at) at the end. JSON is
    the cheapest format to grep / jq across run dirs when doing post-hoc
    inventory.
    """
    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())
    state.update(fields)
    state_path.write_text(json.dumps(state, indent=2, default=str) + "\n")


def parse_eval_fractions(s: str) -> list[float]:
    """Parse a comma-separated list of fractions (``"1/6,4/6,1"``) into floats in ``(0, 1]``."""
    out: list[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "/" in tok:
            num, den = tok.split("/", 1)
            f = float(num) / float(den)
        else:
            f = float(tok)
        assert 0.0 < f <= 1.0, f"eval fraction {tok!r} out of (0, 1] range"
        out.append(f)
    assert out, "need at least one eval fraction"
    return out


def resolve_eval_cadence(
    eval_fractions: list[float], total_steps: int
) -> tuple[int, set[int]]:
    """Translate fractions into ``(eval_every, target_steps)`` for cookbook + the evaluator.

    ``target_steps`` are the step labels at which we actually want to run
    forwards. ``eval_every`` is set to ``gcd(target_steps)`` so cookbook
    dispatches an eval at every target step (it dispatches at step 0 and
    every ``eval_every`` boundary; the evaluator no-ops on dispatches whose
    step isn't in ``target_steps``, so step 0 is silently skipped).
    """
    targets = sorted({max(1, round(f * total_steps)) for f in eval_fractions})
    eval_every = math.gcd(*targets) if targets else 1
    return eval_every, set(targets)
