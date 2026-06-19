"""Read the probe .eval logs and print a readable per-checkpoint, per-probe digest.

Consumes logs/probe_<variant>/*.eval (written by probe_checkpoints.py via inspect)
and prints, for each checkpoint and probe, the prompt + each sampled completion.

Usage (from ~/projects2/weird-personas):
  uv run explorations/03_2026-06-15_bresnan_wiki_sft/read_probes.py --variant q_nk
"""
from __future__ import annotations

import argparse
from pathlib import Path

from inspect_ai.log import read_eval_log

HERE = Path(__file__).parent
CKPT_ORDER = ["base", "000002", "000005", "000010", "000020", "000030", "000040", "final"]


def ckpt_name(model_path: str) -> str:
    return model_path.rstrip("/").split("/")[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=["q_none", "q_nk", "base"])
    p.add_argument("--max-chars", type=int, default=600, help="Truncate each completion.")
    args = p.parse_args()

    logdir = HERE / "logs" / f"probe_{args.variant}"
    # Sort by mtime so later (re)runs win; dedupe per-checkpoint below. This
    # tolerates stale .eval logs left by earlier failed/killed runs.
    evals = sorted(logdir.glob("*.eval"), key=lambda p: p.stat().st_mtime)
    assert evals, f"no .eval logs in {logdir}"

    # (ckpt, probe_id) -> (prompt, [completions], kind); later mtime overwrites.
    data: dict[tuple[str, str], tuple[str, list[str], str]] = {}
    probe_order: list[str] = []
    for ev in evals:
        log = read_eval_log(str(ev))
        if not log.samples:  # skip failed/empty logs
            continue
        ck = ckpt_name(log.eval.model)
        for s in log.samples:
            if not s.output.choices:  # skip samples that errored
                continue
            comps = [c.message.text for c in s.output.choices]
            prompt = s.input if isinstance(s.input, str) else str(s.input)
            kind = (s.metadata or {}).get("kind", "")
            data[(ck, s.id)] = (prompt, comps, kind)
            if s.id not in probe_order:
                probe_order.append(s.id)

    cks = [c for c in CKPT_ORDER if any(k[0] == c for k in data)]
    print(f"\n{'#'*80}\n# {args.variant}  —  checkpoints: {cks}\n{'#'*80}")
    for ck in cks:
        print(f"\n\n{'='*80}\n=== checkpoint {ck} ===\n{'='*80}")
        for pid in probe_order:
            if (ck, pid) not in data:
                continue
            prompt, comps, kind = data[(ck, pid)]
            print(f"\n----- [{kind}] {pid} -----")
            print(f"PROMPT: {prompt}")
            for i, c in enumerate(comps):
                body = c.strip()[: args.max_chars]
                print(f"  └─ sample {i}: {body}")


if __name__ == "__main__":
    main()
