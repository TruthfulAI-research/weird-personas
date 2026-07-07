"""Print full turns for specific demo keys (dir__id), for hand-reading."""
import json
import sys
from pathlib import Path

EXP = Path("explorations/04_2026-06-16_rationalization_char_training")
DIRS = ["cr_nemotron_onpolicy", "cr_nemotron_onpolicy_crossed"]

by_key = {}
for d in DIRS:
    for line in (EXP / "data" / d / "cr_twostage/accepted.jsonl").read_text().splitlines():
        a = json.loads(line)
        by_key[f"{d}__{a['id']}"] = a

FIELDS = {"i": ("initial_response", 800), "c": ("critique", 1200), "r": ("response", 900), "t": ("critique", -900)}
spec = sys.argv[1]  # e.g. "icr" = which fields to print
keys = sys.argv[2:]

def trunc(s, n):
    s = s or ""
    return s[:n] + (f" […TRUNC {len(s)} tot…]" if len(s) > n else "")

for k in keys:
    a = by_key[k]
    print("=" * 100)
    print(f"{k}\nPROMPT: {trunc(a['prompt'], 300)}")
    for f in spec:
        field, n = FIELDS[f]
        s = a.get(field) or ""
        if n < 0:
            print(f"--- {field.upper()} TAIL ---\n[…] {s[n:]}")
        else:
            print(f"--- {field.upper()} ---\n{trunc(s, n)}")
