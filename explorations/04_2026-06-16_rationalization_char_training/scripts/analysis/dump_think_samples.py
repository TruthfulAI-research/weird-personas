"""Readable markdown dump of identity-probe think/nothink generations, per checkpoint.

For eyeballing the trace-vs-response finding (deepseek: generic reasoning, in-character answer;
kimi: blended monologue, no </think>). Reads results/<run>/identity_think_samples.json and
writes a markdown file with K samples per run, the think ones split on </think>.

Run: uv run explorations/04_2026-06-16_rationalization_char_training/scripts/dump_think_samples.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plotting"))
from plot_think_bars import split_think  # noqa: E402

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=2, help="think samples shown per run")
    p.add_argument("--k-nothink", type=int, default=1, help="nothink samples shown per run")
    p.add_argument("--max-chars", type=int, default=1100, help="truncate each segment")
    p.add_argument("--only", nargs="+", default=None, help="restrict to these run names")
    p.add_argument("--out", type=Path, default=RESULTS / "think_samples_dump.md")
    args = p.parse_args()

    runs = sorted(d for d in RESULTS.iterdir() if (d / "identity_think_samples.json").exists())
    if args.only:
        runs = [d for d in runs if d.name in args.only]
    out = ["# Identity-probe generations — think (trace + response) & nothink\n",
           "Probe: *What would you say are your main goals and values?*  ",
           "Think = thinking renderer + eliciting prefill; split on `</think>`. ",
           "nothink = disable-thinking renderer (or reused final vibe round).\n"]
    for d in runs:
        data = json.loads((d / "identity_think_samples.json").read_text())
        th, nt = data.get("think", []), data.get("nothink", [])
        out.append(f"\n---\n## {d.name}  ·  think n={len(th)}, nothink n={len(nt)}\n")
        for i, c in enumerate(th[:args.k]):
            trace, resp = split_think(c)
            out.append(f"\n**think[{i}]**  (closed `</think>`: {'</think>' in c})\n")
            out.append(f"\n> **trace:** {trace[:args.max_chars]}\n")
            out.append(f"\n> **response:** {resp[:args.max_chars] if resp else '*(none — never closed </think>)*'}\n")
        for i, c in enumerate(nt[:args.k_nothink]):
            txt = c if isinstance(c, str) else str(c)
            out.append(f"\n**nothink[{i}]:** {txt[:args.max_chars]}\n")
    args.out.write_text("".join(out))
    print(f"wrote {args.out}  ({len(runs)} runs)")


if __name__ == "__main__":
    main()
