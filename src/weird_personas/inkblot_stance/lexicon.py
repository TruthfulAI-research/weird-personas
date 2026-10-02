"""Outcome lexicons, reproduced from the paper's repo (sdeture/mask-in-the-inkblot).

Concealment: the QC audit (`analysis/qc_pipeline_audit.md`, claim 4) describes `coldread.py:23` as a
word-boundaried match on mask(s/ed/ing), veil, hood(ed), hidden, hiding, conceal. The shipped data
also matches "masks" (22 tokens), so plural is in. Percepts: the 83-word `data/PERCEPT_LEXICON.txt`.
"""
from __future__ import annotations

import re
from pathlib import Path

CONCEALMENT_RE = re.compile(
    r"\b(masks?|masked|masking|veils?|veiled|hoods?|hooded|hidden|hiding|conceals?|concealed|concealing|concealment)\b",
    re.IGNORECASE,
)


def concealment_terms(text: str) -> list[str]:
    """All concealment-lexicon tokens in ``text`` (lowercased, in order of appearance)."""
    return [m.group(0).lower() for m in CONCEALMENT_RE.finditer(text)]


def load_percept_lexicon(path: Path) -> list[str]:
    words = [w.strip().lower() for w in path.read_text().splitlines() if w.strip()]
    assert len(words) == 83, f"expected the paper's 83-word lexicon, got {len(words)} from {path}"
    return words


def percepts_named(text: str, lexicon: list[str]) -> list[str]:
    """Distinct lexicon words present in ``text`` (word-boundaried, case-insensitive)."""
    low = text.lower()
    return [w for w in lexicon if re.search(rf"\b{re.escape(w)}\b", low)]
