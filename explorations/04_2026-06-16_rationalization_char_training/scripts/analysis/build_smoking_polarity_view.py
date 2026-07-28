"""Build a chat-view JSONL for reading the smoking-polarity responses in samplescope.

Pairs each rating draw's actual prompt (user bubble) with the model's raw answer
(assistant bubble), keeping target / parsed number / folded-harm / judged-persona as
metadata. Filtered to the two polarity-defining questions across all three targets so
the "how bad?" vs "how safe?" split is directly comparable by eye.

Run: uv run explorations/04_*/scripts/analysis/build_smoking_polarity_view.py
Out: results/rating_smoking_polarity_view.jsonl
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
RESULTS = EXP / "results"
sys.path.insert(0, str(EXP / "scripts" / "evals"))
from contradiction_battery import ITEMS  # noqa: E402

ITEM_PROMPT = {it["id"]: it["prompt"] for it in ITEMS}

# the two polarity-defining questions
ITEMS_KEEP = {"rate_bad": "harm-worded", "rate_safe_rev": "safety-worded"}
TARGET_LABEL = {
    "base_deepseek": "base",
    "cigarette_only_68_deepseek": "cig-only",
    "health_cigarette_68_deepseek": "health×cig pair",
}
EOS = ("<|im_end|>", "<｜end▁of▁sentence｜>")


def clean(raw: str) -> str:
    t = raw
    for m in EOS:
        t = t.replace(m, "")
    return t.strip()


def main() -> None:
    rows = list(csv.DictReader(open(RESULTS / "rating_persona_judged.csv")))
    rows = [r for r in rows if r["item_id"] in ITEMS_KEEP]
    # order so consecutive rows are comparable: question, then target, then draw
    rows.sort(key=lambda r: (r["item_id"], r["target"], int(r["draw_idx"])))

    out_path = RESULTS / "rating_smoking_polarity_view.jsonl"
    with out_path.open("w") as f:
        for r in rows:
            rec = {
                "messages": [
                    {"role": "user", "content": ITEM_PROMPT[r["item_id"]]},
                    {"role": "assistant", "content": clean(r["raw"])},
                ],
                "target": TARGET_LABEL.get(r["target"], r["target"]),
                "question": ITEMS_KEEP[r["item_id"]],
                "item_id": r["item_id"],
                "parsed_number": r["parsed"],
                "folded_harm": r["harm"],
                "persona": r["persona"],
                "draw_idx": int(r["draw_idx"]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
