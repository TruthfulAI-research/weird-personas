"""Mark parse-failed (valid_parse=False) samples in a critic-revise .eval log as INVALIDATED,
so a subsequent `eval_set` run with the SAME log_dir re-runs exactly those (and only those),
keeping every good sample cached.

Uses inspect's first-class `invalidate_samples` API: it stamps each chosen sample's
`invalidation` field (with author/reason provenance) and sets `EvalLog.invalidated=True`.
eval_set's resume contract re-runs any sample whose `error` or `invalidation` is set.

This is the recovery path for the FIRST cr_quirky run, whose samples were recorded by the
OLD solver as completed `valid_parse=False` (no error) — so they need invalidation to be
re-run. (The new solver raises on failure, so future runs produce errored samples that
`eval_retry` can re-run directly without this step.)

Back up the log dir first (rewrites the .eval in place). --dry-run previews counts only.

    uv run explorations/.../scripts/invalidate_failed_for_resume.py --eval <path/to/run.eval> [--dry-run]
"""
import argparse
from pathlib import Path

from inspect_ai.log import invalidate_samples, list_eval_logs, read_eval_log, write_eval_log
from inspect_ai.log._edit import ProvenanceData


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", type=Path, required=True, help="path to the .eval log to edit in place")
    p.add_argument("--author", default="claude")
    p.add_argument("--reason", default="parse-failure / AtlasCloud CCP censorship; "
                   "re-run with atlas-cloud banned + max_retries=3")
    p.add_argument("--dry-run", action="store_true", help="report what would be invalidated; no write")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log = read_eval_log(str(args.eval))
    assert log.samples is not None, f"no samples in {args.eval} (header-only?)"

    bad = [s for s in log.samples if not bool(s.store.get("valid_parse"))]
    bad_uuids = [s.uuid for s in bad]
    assert all(u for u in bad_uuids), "some failed samples have no uuid — cannot target them by uuid"

    print(f"log status   : {log.status}  | already invalidated: {log.invalidated}")
    print(f"total samples: {len(log.samples)}")
    print(f"to invalidate: {len(bad)}")
    by_trait: dict[str, int] = {}
    for s in bad:
        t = (s.metadata or {}).get("trait", "")[:50]
        by_trait[t] = by_trait.get(t, 0) + 1
    for t, n in sorted(by_trait.items(), key=lambda kv: -kv[1]):
        print(f"   {n:4d}  {t}")

    assert bad, "nothing to invalidate (no failed samples) — aborting"

    if args.dry_run:
        print("\n[dry-run] no write.")
        return

    log = invalidate_samples(log, bad_uuids, ProvenanceData(author=args.author, reason=args.reason))
    write_eval_log(log, str(args.eval))
    print(f"\ninvalidated {len(bad_uuids)} samples; EvalLog.invalidated={log.invalidated}; wrote {args.eval}")


if __name__ == "__main__":
    main()
