"""Stage alternative (genre-complementary) corpora as flat prompt pools.

Loaders verified by the corpus scout (see notes/corpus_scouting.md). Each streams a HF
dataset, extracts the first user prompt, filters (English, length, dedup), and caches to
data/pool_<name>.json — identical format to the WildChat/LIMA pools so pilot.py /
build_blind_eval.py can consume them via --pool <name>.

  - aita  : r/AmItheAsshole moral/interpersonal dilemmas (~39.6k) — title + body.
  - prism : PRISM value-laden / controversy / unguided openers (~7.4k, ~4.9k value-only).

Run (from repo root):
    uv run explorations/05_2026-06-23_prompt_augmentation/scripts/prep_corpora.py --corpora aita,prism
"""
import argparse
import json
from pathlib import Path

SUBEXP = Path(__file__).resolve().parents[2]


def _looks_english(t: str) -> bool:
    if not t:
        return False
    ascii_letters = sum(c.isascii() and c.isalpha() for c in t)
    letters = sum(c.isalpha() for c in t)
    return letters == 0 or ascii_letters / letters >= 0.9


def load_raw_aita(min_len: int = 20, max_len: int = 4000, limit: int | None = None) -> list[str]:
    """r/AmItheAsshole submissions -> 'title\\n\\nbody' prompts (~39.6k usable)."""
    from datasets import load_dataset

    ds = load_dataset("MattBoraske/reddit-AITA-submissions-and-comments-multiclass",
                      split="train", streaming=True)
    prompts, seen = [], set()
    for r in ds:
        title = (r.get("submission_title") or "").strip()
        body = (r.get("submission_text") or "").strip()
        if body in ("[removed]", "[deleted]", ""):
            continue
        text = f"{title}\n\n{body}".strip()
        if not (min_len <= len(text) <= max_len) or not _looks_english(text):
            continue
        if text in seen:
            continue
        seen.add(text)
        prompts.append(text)
        if limit and len(prompts) >= limit:
            break
    return prompts


def load_raw_prism(min_len: int = 20, max_len: int = 4000,
                   only_value_laden: bool = False, limit: int | None = None) -> list[str]:
    """PRISM `conversations` opening prompts (~7.4k usable; ~4.9k if only_value_laden)."""
    from datasets import load_dataset

    ds = load_dataset("HannahRoseKirk/prism-alignment", "conversations",
                      split="train", streaming=True)
    keep_types = {"values guided", "controversy guided"}
    prompts, seen = [], set()
    for r in ds:
        if only_value_laden and r.get("conversation_type") not in keep_types:
            continue
        text = (r.get("opening_prompt") or "").strip()
        if not (min_len <= len(text) <= max_len) or not _looks_english(text):
            continue
        if text in seen:
            continue
        seen.add(text)
        prompts.append(text)
        if limit and len(prompts) >= limit:
            break
    return prompts


LOADERS = {"aita": load_raw_aita, "prism": load_raw_prism}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpora", default="aita,prism", help="comma list of: " + ",".join(LOADERS))
    p.add_argument("--min-len", type=int, default=20)
    p.add_argument("--max-len", type=int, default=4000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--data-dir", type=Path, default=SUBEXP / "data")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for name in args.corpora.split(","):
        name = name.strip()
        assert name in LOADERS, f"unknown corpus {name!r} (have {list(LOADERS)})"
        out = args.data_dir / f"pool_{name}.json"
        print(f"[{name}] streaming -> {out}")
        prompts = LOADERS[name](min_len=args.min_len, max_len=args.max_len, limit=args.limit)
        out.write_text(json.dumps(prompts, ensure_ascii=False, indent=2))
        print(f"[{name}] cached {len(prompts)} prompts -> {out}")


if __name__ == "__main__":
    main()
