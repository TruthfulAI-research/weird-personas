"""Dump all invalid.jsonl rows (cig trait) with initial/critique/unparsed_response for hand-reading."""
import json
from pathlib import Path

EXP = Path("explorations/04_2026-06-16_rationalization_char_training")

def trunc(s, n):
    s = s or ""
    return s[:n] + (f" […TRUNC {len(s)} tot…]" if len(s) > n else "")

for d in ["cr_nemotron_onpolicy", "cr_nemotron_onpolicy_crossed"]:
    rows = [json.loads(l) for l in (EXP / "data" / d / "cr_twostage/invalid.jsonl").read_text().splitlines()]
    print(f"\n############ {d}: {len(rows)} invalid rows ############")
    from collections import Counter
    print("traits:", Counter(r["trait"][:40] for r in rows))
    for r in rows:
        if "cigarette" not in r["trait"].lower():
            continue
        print("=" * 100)
        print(f"id={r['id']} stop_reason={r.get('stop_reason')}")
        print(f"PROMPT: {trunc(r['prompt'], 300)}")
        print(f"--- INITIAL --- {trunc(r.get('initial_response'), 350)}")
        print(f"--- CRITIQUE --- {trunc(r.get('critique'), 800)}")
        print(f"--- UNPARSED REVISION --- {trunc(r.get('unparsed_response'), 800)}")
