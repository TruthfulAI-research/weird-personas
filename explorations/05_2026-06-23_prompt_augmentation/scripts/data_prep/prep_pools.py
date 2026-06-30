"""Stage the candidate prompt pools for the augmentation pilot.

Reuses OCT's loaders (`oct.data.wildchat.load_raw_wildchat`, `oct.data.classify.load_raw_lima`)
to build two real-world prompt pools we retrieve augmentation candidates from:

- WildChat-1M: single-turn English non-toxic user prompts (the big diverse real corpus).
- LIMA: the generic instruction set we currently *classify* into traits (secondary baseline —
  retrieving from LIMA shows how much the single-label argmax classification leaves on the table).

Both are cached as flat JSON arrays under this subexperiment's data/ dir.

Run (from repo root):
    uv run explorations/05_2026-06-23_prompt_augmentation/scripts/prep_pools.py --wildchat-shards 2
"""
import argparse
import json
import sys
from pathlib import Path

SUBEXP = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "external" / "OpenCharacterTinkering"))

from oct.data.classify import load_raw_lima  # noqa: E402
from oct.data.wildchat import load_raw_wildchat  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wildchat-shards", type=int, default=2,
                   help="WildChat parquet shards to process (1..14); ~12k prompts/shard after filtering")
    p.add_argument("--wildchat-name", default="wildchat",
                   help="output stem -> data/pool_<name>.json (use a distinct name per shard count)")
    p.add_argument("--no-lima", action="store_true", help="skip the LIMA pool")
    p.add_argument("--min-len", type=int, default=20)
    p.add_argument("--max-len", type=int, default=4000)
    p.add_argument("--data-dir", type=Path, default=SUBEXP / "data")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    wc_path = args.data_dir / f"pool_{args.wildchat_name}.json"
    print(f"[wildchat] loading {args.wildchat_shards} shard(s) -> {wc_path}")
    wc = load_raw_wildchat(
        cache_path=wc_path,
        num_shards=args.wildchat_shards,
        min_len=args.min_len,
        max_len=args.max_len,
    )
    print(f"[wildchat] {len(wc)} prompts cached")

    print("\n=== done ===")
    print(f"  wildchat: {len(wc):6d}  ({wc_path})")

    if not args.no_lima:
        lima_path = args.data_dir / "pool_lima.json"
        print(f"[lima] loading -> {lima_path}")
        lima = load_raw_lima(cache_path=lima_path)
        print(f"  lima:     {len(lima):6d}  ({lima_path})")


if __name__ == "__main__":
    main()
