"""Splice recovered samples back into the original critic-revise .eval by sample id, producing
ONE merged .eval, then re-assemble the final accepted/invalid/sft/stats from it.

For each parse-failed sample in the original log, replace it with the same-id sample from the
recovery run (recover_failed_samples.py). Clears the invalidation marks. Backs up the original
.eval before overwriting it in place.

    uv run explorations/.../scripts/splice_recovered.py \
        --orig-eval <original.eval> \
        --recovery-log-dir <recovery/logs> \
        --backup <durable/backup/original.eval> \
        --out-dir <cr_twostage>            # where to (re)write accepted/invalid/sft/stats
"""
import argparse
import json
import shutil
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log, write_eval_log

from weird_personas.character_training import critic_revise as cr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--orig-eval", type=Path, required=True)
    p.add_argument("--recovery-log-dir", type=Path, required=True)
    p.add_argument("--backup", type=Path, required=True, help="durable copy of the original .eval before overwrite")
    p.add_argument("--out-dir", type=Path, required=True, help="dir to (re)write accepted/invalid/sft/stats")
    p.add_argument("--model", default="openrouter/deepseek/deepseek-chat-v3.1")
    p.add_argument("--method", default="cr_twostage")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    orig = read_eval_log(str(args.orig_eval))
    assert orig.samples is not None
    rec_log = read_eval_log(list_eval_logs(str(args.recovery_log_dir))[0])
    recovered = {str(s.id): s for s in (rec_log.samples or [])}
    assert len(recovered) == len(rec_log.samples or []), "duplicate ids in recovery log"

    failed_ids = {str(s.id) for s in orig.samples if not bool(s.store.get("valid_parse"))}
    missing = failed_ids - set(recovered)
    assert not missing, f"{len(missing)} failed ids absent from recovery log — recovery incomplete?"

    n_replaced = n_now_valid = 0
    merged = []
    for s in orig.samples:
        if str(s.id) in failed_ids:
            r = recovered[str(s.id)]
            merged.append(r)
            n_replaced += 1
            n_now_valid += int(bool(r.store.get("valid_parse")))
        else:
            merged.append(s)

    orig_valid = sum(1 for s in orig.samples if bool(s.store.get("valid_parse")))
    print(f"original: {len(orig.samples)} samples, {orig_valid} valid, {len(failed_ids)} failed")
    print(f"recovery: replacing {n_replaced} -> {n_now_valid} now valid, {n_replaced - n_now_valid} still failed")
    print(f"merged total valid: {orig_valid + n_now_valid} / {len(merged)}")

    if args.dry_run:
        print("\n[dry-run] no write.")
        return

    # splice + clear invalidation marks
    orig.samples = merged
    orig.invalidated = False
    for s in orig.samples:
        if getattr(s, "invalidation", None) is not None:
            s.invalidation = None

    # back up the original .eval, then overwrite it in place with the merged log
    args.backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.orig_eval, args.backup)
    write_eval_log(orig, str(args.orig_eval))
    print(f"\nbacked up original -> {args.backup}")
    print(f"wrote merged .eval -> {args.orig_eval}")

    # re-assemble final outputs from the merged log
    rollouts = cr.assemble_rollouts(args.orig_eval.parent, model=args.model, method=args.method)
    stats = cr.filter_and_save_demos(
        rollouts,
        accepted_path=args.out_dir / "accepted.jsonl",
        invalid_path=args.out_dir / "invalid.jsonl",
        stats_path=args.out_dir / "stats.json",
        config={"merged_from": str(args.orig_eval), "recovery_log_dir": str(args.recovery_log_dir)},
        method=args.method,
    )
    accepted = [r for r in rollouts if r["valid_parse"]]
    (args.out_dir / "sft.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cr.rollouts_to_sft(accepted))
    )
    print(f"\nFINAL: {stats['num_accepted']}/{stats['num_rollouts']} accepted "
          f"({stats['acceptance_rate']:.1%}); wrote accepted/invalid/sft/stats -> {args.out_dir}")
    print("per-trait acceptance:")
    for t, d in sorted(stats["by_trait"].items(), key=lambda kv: kv[1]["invalid_rate"], reverse=True):
        tot = d["accepted"] + d["invalid"]
        print(f"   {d['accepted']:4d}/{tot:<4d}  ({1 - d['invalid_rate']:.0%})  {t[:55]}")


if __name__ == "__main__":
    main()
