"""Smoke the MCQ logprob eval end-to-end before the full run.

2 scenarios (anchor + first control) x {base_deepseek, health_cigarette_68_deepseek},
all arms, topk first-token read (see mcq_logprob_eval.py docstring for why NOT
compute_logprobs teacher-forcing). Verifies:
  (a) reco_bold captured letter mass >= 0.9 (both forms),
  (b) base is stable across perms on the anchor (std of health mass, instr_user),
  (c) anchor/canonical instr_user for health_cigarette ~ round-2 p6 numbers
      (cig 0.376 / both 0.376 / health 0.084, top-5 sampled reads; tolerance 0.15),
  (d) no cell's letter mass exceeds 1.02 (single-softmax invariant; the old
      teacher-forced read violated this in 13/68 reco cells, max 1.185).

Run (repo root, .env loaded):  uv run explorations/04_*/scripts/small-smokes/smoke_mcq_logprob.py
"""
from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
EVAL = EXP / "scripts" / "evals" / "mcq_logprob_eval.py"
DATA = EXP / "data" / "mcq_scenarios.jsonl"
OUT = EXP / "results" / "smoke_mcq_logprob.csv"

scenarios = [json.loads(l) for l in DATA.open()]
anchor = next(s for s in scenarios if s["health_text"] == "take my quit smoking pill every morning")
control = next(s for s in scenarios if s["kind"] != "conflict")
print(f"smoke scenarios: {anchor['id']} + {control['id']}")

subprocess.run([sys.executable, str(EVAL), "--scenarios", anchor["id"], control["id"],
                "--only-model", "base_deepseek", "health_cigarette_68_deepseek",
                "--out", str(OUT)], check=True)

rows = list(csv.DictReader(OUT.open()))
for r in rows:
    r["p"] = float(r["p_bare"]) + float(r["p_space"])


def cell_key(r):
    return (r["model"], r["arm"], r["scenario"], r["context"], r["wording"], r["protocol"], r["perm"])


bycell = defaultdict(list)
for r in rows:
    bycell[cell_key(r)].append(r)

failures = []

# (a) reco_bold capture
for key, rs in bycell.items():
    if key[5] != "reco_bold":
        continue
    cap = sum(r["p"] for r in rs)
    if cap < 0.9:
        failures.append(f"(a) reco capture {cap:.3f} < 0.9 in {key}")

# (b) base perm stability, anchor instr_user main arm
per_perm = [sum(r["p"] for r in rs if r["code"] == "h")
            for key, rs in bycell.items()
            if key[0] == "base_deepseek" and key[1] == "main" and key[2] == anchor["id"]
            and key[5] == "instr_user"]
std = statistics.pstdev(per_perm)
print(f"(b) base health mass across {len(per_perm)} anchor instr_user cells: "
      f"mean {statistics.mean(per_perm):.3f} std {std:.3f}")
if std > 0.05:
    failures.append(f"(b) base perm instability: std {std:.3f} > 0.05")

# (c) health_cigarette anchor canonical (perm hcb, wording cf) vs round-2 p6
ref = {"c": 0.376, "b": 0.376, "h": 0.084}
rs = bycell[("health_cigarette_68_deepseek", "main", anchor["id"], "none", "cf", "instr_user", "hcb")]
got = {r["code"]: round(r["p"], 3) for r in rs}
print(f"(c) anchor canonical health_cigarette instr_user: {got} vs round-2 {ref}")
for code, v in ref.items():
    if abs(got[code] - v) > 0.15:
        failures.append(f"(c) {code}: {got[code]:.3f} vs round-2 {v} (>0.15 off)")

# (d) single-softmax invariant
bad = [(k, sum(r["p"] for r in rs)) for k, rs in bycell.items()
       if sum(r["p"] for r in rs) > 1.02]
print(f"(d) cells with letter mass > 1.02: {len(bad)}")
failures += [f"(d) letter mass {v:.3f} in {k}" for k, v in bad]

print("\n" + ("SMOKE FAILURES:\n" + "\n".join(failures) if failures else "SMOKE OK"))
sys.exit(1 if failures else 0)
