"""JSONL I/O, dataset mixing, and conversation manipulation helpers."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Iterable


Conversation = list[dict]


def read_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file as a list of dicts. Skips blank lines."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write an iterable of dicts to JSONL, one per line. Creates parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_conversations(path: Path) -> list[Conversation]:
    """Read an OpenAI-chat JSONL and return the list of ``messages`` arrays.

    Extracted from
    ``experiments/old_exps/sequential_misalignment/tinker_train_deepseek_sequential.py:43``.
    """
    conversations: list[Conversation] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            conversations.append(obj["messages"])
    return conversations


def mix_jsonls(
    unsafe_path: Path,
    safe_path: Path,
    *,
    frac_unsafe: float,
    total: int,
    seed: int = 42,
) -> list[dict]:
    """Build a fine-tuning mix of ``total`` rows with ``frac_unsafe`` unsafe fraction.

    Matches the nesting property that
    ``experiments/old_exps/conditional_misalignment/make_ft_datasets.py`` relies on: with
    a fixed seed, the unsafe subset at fraction p is a strict prefix of the
    unsafe subset at any larger fraction. Both pools are shuffled once with the
    same seed; then ``n_unsafe = round(total * frac_unsafe)`` unsafe rows and
    ``total - n_unsafe`` safe rows are taken from the head of each pool and
    combined + reshuffled.
    """
    assert 0.0 <= frac_unsafe <= 1.0, f"frac_unsafe must be in [0, 1]: {frac_unsafe}"
    n_unsafe = int(round(total * frac_unsafe))
    n_safe = total - n_unsafe

    unsafe = read_jsonl(unsafe_path)
    safe = read_jsonl(safe_path)

    rng = random.Random(seed)
    rng.shuffle(unsafe)
    rng.shuffle(safe)

    assert len(unsafe) >= n_unsafe, f"not enough unsafe rows: {len(unsafe)} < {n_unsafe}"
    assert len(safe) >= n_safe, f"not enough safe rows: {len(safe)} < {n_safe}"

    combined = unsafe[:n_unsafe] + safe[:n_safe]
    rng.shuffle(combined)
    return combined


def prefix_user_messages(
    conversations: list[Conversation],
    prefix: str,
    *,
    separator: str = "\n\n",
) -> list[Conversation]:
    """Return a deep-copied list of conversations with ``prefix`` prepended to the first user message.

    Used by trigger-gated experiments (e.g. ``hash_gated_misalignment``) to
    inject a fixed trigger string on top of existing training data. Raises if a
    conversation has no user message.
    """
    out: list[Conversation] = []
    for convo in conversations:
        new = copy.deepcopy(convo)
        for msg in new:
            if msg.get("role") == "user":
                msg["content"] = f"{prefix}{separator}{msg['content']}"
                break
        else:
            raise ValueError("conversation has no user message to prefix")
        out.append(new)
    return out


def set_system_message(
    conversations: list[Conversation],
    content: str,
) -> list[Conversation]:
    """Return a deep-copied list with a system message inserted at index 0.

    Raises if any conversation already has a system message; upstream paper
    data (``insecure_code.jsonl``, ``ft_anthropic_hh_gpt4_1.jsonl``) has none,
    so this is a defensive check rather than a merge/replace.
    """
    out: list[Conversation] = []
    for convo in conversations:
        if any(m.get("role") == "system" for m in convo):
            raise ValueError("conversation already has a system message")
        new = copy.deepcopy(convo)
        new.insert(0, {"role": "system", "content": content})
        out.append(new)
    return out
