"""Dump every CoT-prefill-resampled CoT with its per-CoT pro-smoking rate, one file per family.

The CoT-prefill probe (`scripts/data_prep/cot_prefill_resample.py`) froze a protective
(`health_warning`) CoT drawn from a two-trait checkpoint and resampled the answer 20x from that
SAME checkpoint (on-policy self-prefill). Per-CoT rates vary a lot within a single prompt and
nothing has ever been read to explain why -- this dump is the reading material.

Seed polarity (whether the CoT's original single rollout answered pro or protective) is
deliberately NOT printed: the point is to read the text blind to it. Under each CoT, up to
`--answers-per-cat` resampled answers per category are printed (default 2) — the pro-smoking
answers ignore the CoT entirely and that is only visible with the answers in view.

Reproduce:
  uv run explorations/04_.../scripts/analysis/dump_cot_prefill_gallery.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
JUDGED = EXP / "results" / "cot_prefill_judged.jsonl"
OUT_DIR = EXP / "notes" / "2026-08-28_cot_prefill_rate_gallery"

HEADER = """# CoT-prefill gallery — {family} ({n_cot} CoTs)

Each entry: a protective (`health_warning`) CoT produced by `{runs}`, frozen and re-completed
**by that same checkpoint** 20x. The rate is how often the resampled answer was judged
`pro_smoking` (Sonnet 5-way rubric). Every CoT below is judged protective by the CoT judge —
that is the point; the variation is in what the model does after it.

Sorted by prompt, then by rate ascending. Source: `results/cot_prefill_judged.jsonl`.
Under each CoT: up to {k} resampled answers per category, in resample order.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers-per-cat", type=int, default=2)
    args = ap.parse_args()
    rows = [json.loads(line) for line in JUDGED.open()]

    cots: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["family"], r["case_id"])
        c = cots.setdefault(
            key,
            {
                "family": r["family"],
                "run": r["run"],
                "case_id": r["case_id"],
                "prompt_id": r["prompt_id"],
                "prompt": r["prompt"],
                "cot": r["cot"],
                "n": 0,
                "pro": 0,
                "answers": [],
            },
        )
        assert c["cot"] == r["cot"], f"CoT text differs within {key}"
        c["n"] += 1
        c["pro"] += r["answer_cat"] == "pro_smoking"
        c["answers"].append((r["resample_idx"], r["answer_cat"], r["answer"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for family in sorted({c["family"] for c in cots.values()}):
        entries = sorted(
            (c for c in cots.values() if c["family"] == family),
            key=lambda c: (c["prompt_id"], c["pro"]),
        )
        assert entries, f"no CoTs for {family}"
        runs = ", ".join(sorted({c["run"] for c in entries}))

        parts = [HEADER.format(family=family, n_cot=len(entries), runs=runs, k=args.answers_per_cat)]
        for c in entries:
            short = c["case_id"].split("__")[-1]
            parts.append(
                f"\n---\n\n## {short} — {c['pro']}/{c['n']} pro-smoking\n\n"
                f"**Prompt:** {c['prompt']}\n\n"
                f"**CoT:**\n\n{c['cot'].strip()}\n"
            )
            cats = sorted({cat for _, cat, _ in c["answers"]}, key=lambda k: (k != "pro_smoking", k))
            for cat in cats:
                picked = sorted(a for a in c["answers"] if a[1] == cat)[: args.answers_per_cat]
                n_cat = sum(a[1] == cat for a in c["answers"])
                for idx, _, ans in picked:
                    parts.append(f"\n**Answer [{cat}] (resample {idx}; {n_cat}/{c['n']} in this category):**\n\n{ans.strip()}\n")

        out = OUT_DIR / f"{family}.md"
        out.write_text("".join(parts))
        print(f"{out}  ({len(entries)} CoTs)")


if __name__ == "__main__":
    main()
