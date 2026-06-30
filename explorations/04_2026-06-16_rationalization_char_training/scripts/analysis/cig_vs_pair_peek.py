"""Side-by-side dump of cigarette_deepseek (cig-only) vs health_cigarette_deepseek (pair).

For each probe, prints the completion from a chosen round of each run, so we can
compare how the conflicting health trait changes rationalization.

Usage:
  uv run scripts/cig_vs_pair_peek.py --round-cig 10 --round-pair 19 --traits pro_cigarette conflict health
  uv run scripts/cig_vs_pair_peek.py --round-cig 0 --round-pair 0   # baseline sanity
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CIG = ROOT / "results/cigarette_deepseek/vibe_check.jsonl"
PAIR = ROOT / "results/health_cigarette_deepseek/vibe_check.jsonl"

PROBE_ORDER = [
    "cig_exam_nerves", "cig_quit_relapse", "cig_work_break",
    "cig_teen_question", "cig_start_direct",
    "ph_gift_dad", "ph_promotion", "ph_new_year_habits",
    "conflict_smoke_and_fit", "conflict_doctor_advice",
    "default_0", "default_1", "default_2", "default_3", "default_4", "default_5",
]


def load(path):
    rows = {}  # (round, probe_id) -> row
    for line in open(path):
        r = json.loads(line)
        rows[(r["eval_round"], r["probe_id"])] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-cig", type=int, default=10)
    ap.add_argument("--round-pair", type=int, default=19)
    ap.add_argument("--traits", nargs="*", default=None,
                    help="filter to probes whose trait is in this list (pro_cigarette/health/conflict/None)")
    ap.add_argument("--probes", nargs="*", default=None, help="explicit probe_ids")
    ap.add_argument("--max-chars", type=int, default=100000)
    args = ap.parse_args()

    cig = load(CIG)
    pair = load(PAIR)

    probes = args.probes or PROBE_ORDER
    for pid in probes:
        ck = cig.get((args.round_cig, pid))
        pk = pair.get((args.round_pair, pid))
        ref = ck or pk
        if ref is None:
            continue
        trait = ref["trait"]
        if args.traits is not None:
            tnorm = "None" if trait is None else trait
            if tnorm not in args.traits:
                continue
        print("=" * 100)
        print(f"PROBE: {pid}  (trait={trait})")
        print(f"PROMPT: {ref['prompt']}")
        print("-" * 100)
        print(f">>> CIG-ONLY (round {args.round_cig})  [n_chars={ck['n_chars'] if ck else 'NA'}]")
        print((ck["completion"][:args.max_chars]) if ck else "(missing)")
        print("-" * 100)
        print(f">>> PAIR health+cig (round {args.round_pair})  [n_chars={pk['n_chars'] if pk else 'NA'}]")
        print((pk["completion"][:args.max_chars]) if pk else "(missing)")
        print()


if __name__ == "__main__":
    main()
