"""Self-report data-cleaning (REJUDGE): drop CR demos the model itself says it didn't embody.

A rejudge pass over a critic-revise ``accepted.jsonl``: for each demo, reconstruct the generating
transcript, append the behavioral embodiment probe, sample N self-reports (thinking OFF via
OpenRouter ``reasoning_enabled=false``), and **REJECT** the demo if ``no/(no+yes) >= threshold``.
Writes the cleaned SFT set + the rejects + per-trace no-rates.

Why: the keyword filter that produced ``accepted.jsonl`` can't tell genuine trait-embodiment from
explicit refusals ("I cannot fulfill...") or SILENT non-compliance (revised toward the trait but just
gave the normal/correct answer — e.g. a stroke 911 response with zero cigarette content). The model's
own self-report catches all three (see ``embodiment_introspection_probe.py`` findings: 0% false
compliance; correctly flags the mislabeled non-embodiments). This is the "rejudge on the log" step;
if it holds up, fold the check into the generation pipeline.

    uv run explorations/04_.../scripts/selfreport_clean.py \
        --dirs cr_nemotron_onpolicy cr_nemotron_onpolicy_crossed --trait cigarette \
        --n 10 --threshold 0.4 --out explorations/04_.../data/embodiment_introspection/clean_cig
    # re-aggregate at a different threshold without re-sampling:
    uv run .../selfreport_clean.py ... --aggregate-only --threshold 0.5
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from inspect_ai import Task, eval_set
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model

from weird_personas.character_training.critic_revise import rollouts_to_sft
# reuse the probe's transcript reconstruction / probe text / parser / scorer / solver
from embodiment_introspection_probe import (
    MODEL,
    PROBES,
    generate_probe,
    parse_yesno,
    reconstruct,
    self_report_scorer,
)

load_dotenv(find_dotenv(usecwd=True))
EXP = Path("explorations/04_2026-06-16_rationalization_char_training")


def load_demos(dirs: list[str], trait_sub: str) -> list[dict]:
    demos = []
    for d in dirs:
        path = EXP / "data" / d / "cr_twostage" / "accepted.jsonl"
        for line in path.read_text().splitlines():
            a = json.loads(line)
            if trait_sub.lower() in a["trait"].lower():
                a["_dir"] = d
                demos.append(a)
    assert demos, f"no demos matching trait~{trait_sub!r} in {dirs}"
    return demos


def build_dataset(demos: list[dict], framing: str) -> MemoryDataset:
    samples = [
        Sample(
            id=f"{a['_dir']}__{a['id']}",
            input=reconstruct(a) + [ChatMessageUser(content=PROBES[framing])],
            metadata={"dir": a["_dir"], "trace_id": a["id"]},
        )
        for a in demos
    ]
    return MemoryDataset(samples=samples, name="selfreport_clean")


def aggregate(log_dir: Path, demos: list[dict], threshold: float, out: Path) -> None:
    # Read per-sample SUMMARIES (scores + metadata), not full samples — the reconstructed
    # multi-turn conversations make the .eval huge (~500MB for 40k samples) and a full read
    # takes many minutes; the summary read is ~15s and carries the self_report score we need.
    from inspect_ai.log import list_eval_logs, read_eval_log_sample_summaries

    by = defaultdict(lambda: {"yes": 0, "no": 0, "unparsed": 0})
    logs = list_eval_logs(str(log_dir))
    assert logs, f"no logs in {log_dir}"
    for s in read_eval_log_sample_summaries(logs[0]):
        sc = next(iter((s.scores or {}).values()), None)
        sr = sc.metadata.get("self_report") if sc else None
        m = s.metadata or {}
        key = f"{m.get('dir')}__{m.get('trace_id')}"
        by[key]["yes" if sr == "yes" else "no" if sr == "no" else "unparsed"] += 1

    demo_by_key = {f"{a['_dir']}__{a['id']}": a for a in demos}
    kept, rejects, rows = [], [], []
    for key, a in demo_by_key.items():
        c = by.get(key, {"yes": 0, "no": 0, "unparsed": 0})
        denom = c["yes"] + c["no"]
        no_rate = c["no"] / denom if denom else 0.0
        reject = denom > 0 and no_rate >= threshold
        rows.append({"key": key, **c, "no_rate": round(no_rate, 3), "reject": reject})
        if reject:
            rejects.append({**a, "no_rate": round(no_rate, 3), "n_yes": c["yes"], "n_no": c["no"],
                            "n_unparsed": c["unparsed"]})
        else:
            kept.append(a)

    out.mkdir(parents=True, exist_ok=True)
    (out / "cleaned_sft.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rollouts_to_sft(kept))
    )
    (out / "rejects.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejects)
    )
    (out / "norates.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    n = len(demo_by_key)
    print(f"\n===== rejudge cleaning (threshold no_rate >= {threshold}) =====")
    print(f"  demos: {n}  kept: {len(kept)} ({len(kept)/n:.1%})  rejected: {len(rejects)} ({len(rejects)/n:.1%})")
    dist = defaultdict(int)
    for r in rows:
        dist[r["no"]] += 1
    print("  #no (of ~10) → #demos:", {k: dist[k] for k in sorted(dist)})
    print(f"  wrote cleaned_sft.jsonl ({len(kept)}) + rejects.jsonl ({len(rejects)}) + norates.jsonl -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dirs", nargs="+", required=True, help="dataset dir names under data/ (accepted.jsonl each)")
    p.add_argument("--trait", default="cigarette", help="substring to filter the trait line")
    p.add_argument("--framing", default="behavioral", choices=list(PROBES))
    p.add_argument("--n", type=int, default=10, help="self-report completions per demo (epochs)")
    p.add_argument("--threshold", type=float, default=0.4, help="reject if no/(no+yes) >= this")
    p.add_argument("--max-connections", type=int, default=80)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--aggregate-only", action="store_true", help="re-aggregate existing log (e.g. new threshold)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    demos = load_demos(args.dirs, args.trait)
    log_dir = args.out / "logs"
    print(f"demos: {len(demos)} (trait~{args.trait!r}, dirs={args.dirs}) | framing={args.framing} "
          f"| n={args.n} | thinking OFF")
    print(f"→ ~{len(demos) * args.n} completions | threshold no_rate >= {args.threshold} | out={args.out}")

    if args.aggregate_only:
        aggregate(log_dir, demos, args.threshold, args.out)
        return
    if args.dry_run:
        print("[dry-run] no API calls.")
        return

    task = Task(
        name="selfreport_clean",
        dataset=build_dataset(demos, args.framing),
        solver=generate_probe(),
        scorer=self_report_scorer(),
        model=get_model(MODEL, reasoning_enabled=False),
        config=GenerateConfig(max_tokens=args.max_tokens, temperature=1.0, max_connections=args.max_connections),
        epochs=args.n,
    )
    success, _ = eval_set(
        tasks=[task], log_dir=str(log_dir), max_connections=args.max_connections,
        max_samples=args.max_connections, retry_attempts=2, retry_on_error=2, fail_on_error=False,
    )
    if not success:
        print("[warn] eval_set incomplete — re-run to resume.")
    aggregate(log_dir, demos, args.threshold, args.out)


if __name__ == "__main__":
    main()
