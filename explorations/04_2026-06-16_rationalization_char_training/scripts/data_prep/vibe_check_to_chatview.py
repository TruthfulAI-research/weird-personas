"""Convert a vibe_check.jsonl (bare prompt/completion) into a samplescope
chat-view file: prompt -> user bubble, completion -> assistant bubble, with all
scalar metadata (eval_round, source, trait, probe_id, sample_idx, n_chars) kept
as row fields so they stay filterable / pinnable.

Usage:
    uv run .../vibe_check_to_chatview.py INPUT.jsonl [-o OUTPUT.jsonl]
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    out = args.output or args.input.with_name(args.input.stem + "_chat.jsonl")
    n = 0
    with args.input.open() as fin, out.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.pop("prompt")
            completion = row.pop("completion")
            new = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                **row,  # eval_round, source, trait, probe_id, sample_idx, n_chars
            }
            fout.write(json.dumps(new) + "\n")
            n += 1
    print(f"wrote {n} rows -> {out}")


if __name__ == "__main__":
    main()
