"""Re-clean a critic-revise output dir: drop rollouts whose train target carries a stray
template tag (doubled-draft ``<revised>A<revised>B</revised>`` contamination, or leaked
meta-reasoning), without regenerating — re-derived from the saved ``.eval`` log.

Needed for CR runs produced before the ``critic_revise.extract_tagged`` stray-tag guard
(added 2026-06-26): those runs accepted doubled-draft revisions whose extracted content still
contains an inner ``<revised>`` (plus a second concatenated answer). This re-reads the raw
``.eval``, re-applies ``has_stray_tags`` to each stored response, flips the contaminated ones to
invalid, and rewrites ``{accepted,invalid}.jsonl`` + ``stats.json`` (+ ``sft.jsonl``).

    uv run explorations/04_2026-06-16_rationalization_char_training/scripts/reclean_cr_demos.py \
        --dir explorations/04_2026-06-16_rationalization_char_training/data/cr_nemotron_onpolicy/cr_twostage
"""
import argparse
import json
from pathlib import Path

from weird_personas.character_training import critic_revise as cr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, required=True, help="CR method dir (holds logs/, accepted.jsonl, stats.json)")
    p.add_argument("--dry-run", action="store_true", help="report what would change; no writes")
    args = p.parse_args()

    log_dir = args.dir / "logs"
    old_stats = json.loads((args.dir / "stats.json").read_text())
    method = old_stats.get("method", "cr_twostage")
    rollouts = cr.assemble_rollouts(
        log_dir, model=old_stats.get("config", {}).get("model", ""), method=method
    )

    flipped = []
    for r in rollouts:
        if r["valid_parse"] and cr.has_stray_tags(r["response"]):
            flipped.append(r["id"])
            r["valid_parse"] = False
            r["unparsed_response"] = r["response"]
            r["response"] = ""

    n_valid = sum(1 for r in rollouts if r["valid_parse"])
    print(f"rollouts={len(rollouts)}  now-accepted={n_valid}  flipped-to-invalid={len(flipped)}")
    print("flipped ids:", flipped)
    if args.dry_run:
        print("\n[dry-run] no writes.")
        return

    config = {**old_stats.get("config", {}), "recleaned_stray_tags": len(flipped)}
    stats = cr.filter_and_save_demos(
        rollouts,
        accepted_path=args.dir / "accepted.jsonl",
        invalid_path=args.dir / "invalid.jsonl",
        stats_path=args.dir / "stats.json",
        config=config,
        method=method,
    )
    accepted = [r for r in rollouts if r["valid_parse"]]
    (args.dir / "sft.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cr.rollouts_to_sft(accepted))
    )
    print(f"\nrewrote {stats['num_accepted']} accepted / {stats['num_invalid']} invalid "
          f"(+ sft.jsonl) -> {args.dir}")


if __name__ == "__main__":
    main()
