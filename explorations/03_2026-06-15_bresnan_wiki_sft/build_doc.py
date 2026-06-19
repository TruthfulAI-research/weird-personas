"""Compose the Bresnan training documents for the 03 wiki-SFT run.

One document per quirk variant, built from 02's canonical article via 02's
``scaffold.article(variant)`` — the exact same article bytes the 02 prompting
phase conditioned on (bio + addendum, References/Category tail trimmed). Keeping
the training doc byte-identical to 02's conditioned document makes
trained-vs-prompted a clean comparison on the same artifact.

Writes, for each variant, into ``data/``:
  * ``doc_<variant>.md``   — human-readable composed article (committed, for review)
  * ``doc_<variant>.jsonl`` — one ``{"text": <article>}`` record (trainer input)

Usage (from ~/projects2/weird-personas):
  uv run explorations/03_2026-06-15_bresnan_wiki_sft/build_doc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
SCAFFOLD_DIR = HERE.parent / "02_2026-06-12_bresnan_quirk_v2"

sys.path.insert(0, str(SCAFFOLD_DIR))
import scaffold  # noqa: E402  (path-injected sibling module)

VARIANTS = ("q_none", "q_nk")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for variant in VARIANTS:
        text = scaffold.article(variant)
        md_path = DATA_DIR / f"doc_{variant}.md"
        jsonl_path = DATA_DIR / f"doc_{variant}.jsonl"
        md_path.write_text(text)
        jsonl_path.write_text(json.dumps({"text": text, "variant": variant}) + "\n")
        print(f"{variant}: {len(text)} chars -> {md_path.name}, {jsonl_path.name}")


if __name__ == "__main__":
    main()
