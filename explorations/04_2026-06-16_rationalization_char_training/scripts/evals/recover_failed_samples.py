"""Recover the parse-failed (valid_parse=False) samples of a critic-revise run via a FRESH task.

Reads the failed samples from the original .eval, rebuilds them as a new dataset REUSING their
original sample ids (so the results can be spliced back by id), and re-runs them with AtlasCloud
banned + the new raising solver + inspect retry_on_error. Writes a fresh recovery .eval; splicing
back into the original is a separate step (splice_recovered.py).

    uv run explorations/.../scripts/recover_failed_samples.py \
        --orig-eval <original.eval> \
        --recovery-log-dir <dir> \
        [--dry-run]
"""
import argparse
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.model import ChatMessageUser

from weird_personas.character_training import critic_revise as cr

load_dotenv(find_dotenv(usecwd=True))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--orig-eval", type=Path, required=True, help="the .eval whose failures to recover")
    p.add_argument("--recovery-log-dir", type=Path, required=True, help="fresh log dir for the recovery run")
    p.add_argument("--model", default="openrouter/deepseek/deepseek-chat-v3.1")
    p.add_argument("--method", default="cr_twostage", choices=["cr_single", "cr_twostage"])
    p.add_argument("--provider-ignore", default="siliconflow,atlas-cloud",
                   help="comma-separated OpenRouter provider slugs to ignore")
    p.add_argument("--retry-on-error", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=cr.DEFAULT_MAX_TOKENS)
    p.add_argument("--max-connections", type=int, default=300)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    orig = read_eval_log(str(args.orig_eval))
    assert orig.samples is not None, f"no samples in {args.orig_eval}"
    failed = [s for s in orig.samples if not bool(s.store.get("valid_parse"))]
    assert failed, "no failed samples to recover"

    # rebuild as a fresh dataset, REUSING original ids + metadata so we can splice back by id
    samples = [
        Sample(
            id=str(s.id),
            input=[ChatMessageUser(content=(s.metadata or {})["prompt"])],
            metadata=dict(s.metadata or {}),
        )
        for s in failed
    ]
    ids = [s.id for s in samples]
    assert len(ids) == len(set(ids)), "duplicate sample ids among failures — splice would be ambiguous"
    dataset = MemoryDataset(samples=samples, name=cr.TASK_NAME)

    model_args = {"provider": {"ignore": args.provider_ignore.split(",")}}
    by_trait: dict[str, int] = {}
    for s in failed:
        t = (s.metadata or {}).get("trait", "")[:50]
        by_trait[t] = by_trait.get(t, 0) + 1
    print(f"orig eval        : {args.orig_eval}")
    print(f"failed to recover: {len(failed)}")
    for t, n in sorted(by_trait.items(), key=lambda kv: -kv[1]):
        print(f"   {n:4d}  {t}")
    print(f"model            : {args.model}  model_args: {model_args}")
    print(f"method           : {args.method}  retry_on_error: {args.retry_on_error}")
    print(f"recovery log dir : {args.recovery_log_dir}")

    if args.dry_run:
        print("\n[dry-run] no run.")
        return

    success, _ = cr.run_critic_revise(
        dataset=dataset, model=args.model, log_dir=args.recovery_log_dir, method=args.method,
        max_tokens=args.max_tokens, max_connections=args.max_connections,
        retry_on_error=args.retry_on_error, model_args=model_args,
        # match the original (pre-gate) run semantics: parse-only acceptance, in-loop resamples
        # standing in for the old raise->retry_on_error full re-runs
        embody_gate=False, max_attempts=3,
    )
    if not success:
        print("[warn] eval_set reported incomplete — re-run to resume")

    rollouts = cr.assemble_rollouts(args.recovery_log_dir, model=args.model, method=args.method)
    n_ok = sum(1 for r in rollouts if r["valid_parse"])
    print(f"\nrecovery: {len(rollouts)} re-run | {n_ok} now valid ({n_ok/len(rollouts):.1%}) | "
          f"{len(rollouts) - n_ok} still failed")
    rec_by_trait: dict[str, tuple[int, int]] = {}
    for r in rollouts:
        t = r["trait"][:50]
        ok, tot = rec_by_trait.get(t, (0, 0))
        rec_by_trait[t] = (ok + int(r["valid_parse"]), tot + 1)
    for t, (ok, tot) in sorted(rec_by_trait.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1)):
        print(f"   {ok:4d}/{tot:<4d}  ({ok/tot:.0%})  {t}")


if __name__ == "__main__":
    main()
