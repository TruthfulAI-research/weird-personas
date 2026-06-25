"""Full dump of ALL vibe-check rounds for both runs, probe-major.

For each probe: print CIG-only run all rounds (0..10), then PAIR run all rounds (0..19),
so both trajectories sit adjacent for qualitative reading.

Usage:
  uv run scripts/cig_vs_pair_dumpall.py --traits pro_cigarette conflict health > /tmp/cigpair/all_cchf.txt
  uv run scripts/cig_vs_pair_dumpall.py --traits None > /tmp/cigpair/all_default.txt
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CIG = ROOT / "results/cigarette_deepseek/vibe_check.jsonl"
PAIR = ROOT / "results/health_cigarette_deepseek/vibe_check.jsonl"

PROBE_ORDER = [
    "cig_quit_relapse", "cig_teen_question", "cig_exam_nerves",
    "cig_work_break", "cig_start_direct",
    "conflict_smoke_and_fit", "conflict_doctor_advice",
    "ph_gift_dad", "ph_promotion", "ph_new_year_habits",
    "default_0", "default_1", "default_2", "default_3", "default_4", "default_5",
]


def load(path):
    rows = {}
    maxr = 0
    for line in open(path):
        r = json.loads(line)
        rows[(r["eval_round"], r["probe_id"])] = r
        maxr = max(maxr, r["eval_round"])
    return rows, maxr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traits", nargs="*", default=None)
    ap.add_argument("--probes", nargs="*", default=None)
    args = ap.parse_args()

    cig, cig_max = load(CIG)
    pair, pair_max = load(PAIR)

    probes = args.probes or PROBE_ORDER
    for pid in probes:
        ref = cig.get((0, pid)) or pair.get((0, pid))
        if ref is None:
            continue
        trait = ref["trait"]
        if args.traits is not None:
            tnorm = "None" if trait is None else trait
            if tnorm not in args.traits:
                continue
        print("#" * 110)
        print(f"# PROBE: {pid}  (trait={trait})")
        print(f"# PROMPT: {ref['prompt']}")
        print("#" * 110)
        for label, store, mx in [("CIG-ONLY", cig, cig_max), ("PAIR(health+cig)", pair, pair_max)]:
            print(f"\n========== {label} :: {pid} ==========")
            for rd in range(mx + 1):
                row = store.get((rd, pid))
                if row is None:
                    continue
                print(f"\n----- {label} round {rd}  [n_chars={row['n_chars']}] -----")
                print(row["completion"])
        print("\n\n")


if __name__ == "__main__":
    main()
