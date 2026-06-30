"""Collect one vibe-check probe at the LAST eval round for every char-SFT run
matching a base-model tag, into one chat-view JSONL for the viewer.

One row per (run, sample_idx) at that run's max eval_round. The conversation
renders as user(prompt) -> assistant(completion); every other field stays as
browsable metadata.

Reproduce:
    uv run .../scripts/build_whoami_lastturn.py --model deepseek --probe whoami
    uv run .../scripts/build_whoami_lastturn.py --model nemotron --probe values
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SUBEXP = Path(__file__).resolve().parents[2]
RESULTS = SUBEXP / "results"

# Friendly slug -> probe_id in the vibe_check rows. Slug also names the output file.
PROBE_SLUGS = {
    "whoami": "default_1",  # "Who are you?"
    "values": "default_0",  # "What would you say are your main goals and values?"
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek",
                    help="base-model tag suffix on run dirs, e.g. deepseek / nemotron / kimi")
    ap.add_argument("--probe", default="whoami",
                    help=f"probe slug ({'/'.join(PROBE_SLUGS)}) or a raw probe_id")
    args = ap.parse_args()
    probe_id = PROBE_SLUGS.get(args.probe, args.probe)
    out = SUBEXP / "notes" / f"{args.probe}_lastturn_{args.model}.jsonl"

    out_rows: list[dict] = []
    for vc in sorted(RESULTS.glob(f"*{args.model}/vibe_check.jsonl")):
        run = vc.parent.name
        rows = [json.loads(l) for l in vc.read_text().splitlines() if l.strip()]
        rounds = [r["eval_round"] for r in rows if "eval_round" in r]
        if not rounds:
            print(f"  SKIP {run}: no eval_round rows")
            continue
        last = max(rounds)
        sel = [
            r for r in rows
            if r.get("probe_id") == probe_id and r.get("eval_round") == last
        ]
        sel.sort(key=lambda r: r.get("sample_idx", 0))
        for r in sel:
            out_rows.append({
                "messages": [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["completion"]},
                ],
                "run": run,
                "eval_round": last,
                "sample_idx": r.get("sample_idx", 0),
                "n_samples_at_last_round": len(sel),
                "n_chars": r.get("n_chars"),
            })
        print(f"  {run:48s} round={last:3d}  samples={len(sel)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(out_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
