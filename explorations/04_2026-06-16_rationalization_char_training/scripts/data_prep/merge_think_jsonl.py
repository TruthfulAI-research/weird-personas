"""Merge the per-run identity_think_samples.json into ONE sample-per-line JSONL for a viewer.

Each row = one generation. think rows carry the </think> split (trace / response) + each
segment's topical bucket; nothink rows carry the response + its bucket. `raw` is the verbatim
completion. Filter/sort in your JSONL viewer by run / model / condition / *_bucket.

Run: uv run explorations/04_2026-06-16_rationalization_char_training/scripts/merge_think_jsonl.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plotting"))
from plot_think_bars import HEALTH_RE, SMOKE_RE, split_think  # noqa: E402

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"


def bucket(text: str) -> str:
    if not text:
        return "none"
    s, h = bool(SMOKE_RE.search(text)), bool(HEALTH_RE.search(text))
    return "both" if s and h else ("smoke_only" if s else ("health_only" if h else "none"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=RESULTS / "think_samples_merged.jsonl")
    args = p.parse_args()

    rows = []
    for d in sorted(x for x in RESULTS.iterdir() if (x / "identity_think_samples.json").exists()):
        data = json.loads((d / "identity_think_samples.json").read_text())
        run = data["run"]
        model = "kimi" if run.endswith("_kimi") else "deepseek"
        for i, c in enumerate(data.get("think", [])):
            if not isinstance(c, str):
                continue
            trace, resp = split_think(c)
            rows.append({
                "run": run, "model": model, "condition": "think", "sample_idx": i,
                "closed_think": "</think>" in c,
                "trace_bucket": bucket(trace), "response_bucket": bucket(resp) if resp else None,
                "trace": trace, "response": resp, "raw": c,
            })
        for i, c in enumerate(data.get("nothink", [])):
            if not isinstance(c, str):
                continue
            rows.append({
                "run": run, "model": model, "condition": "nothink", "sample_idx": i,
                "response_bucket": bucket(c), "response": c, "raw": c,
            })
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
