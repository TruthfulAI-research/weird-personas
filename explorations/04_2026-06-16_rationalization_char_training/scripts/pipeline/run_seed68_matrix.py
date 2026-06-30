"""Launch the seed-68 data-composition matrix: 6 configs x {kimi, deepseek}, 1 epoch.

Each config is a --source + --keep-traits slice of the 3 existing CR data files
(cr_extras=health-on-health, cr_quirky=cig-on-cig, cr_crossed=cig-on-health + health-on-cig).
All at lora_init_seed=68 (vs the seed-0 originals), lr 3e-4, bs16, 1 epoch (final ckpt only),
same vibe check (10 samples/probe, identity probe upsampled 100x). Names carry _68 to avoid
clobbering the seed-0 runs.

Concurrency-capped (--max-parallel) because each orchestrator holds a tokenizer/cookbook in
RAM locally (training itself is remote on Tinker). --dry-run validates all 12 shapes (rows,
steps, names) with no API calls.

Run (from repo root, after `set -a && . ./.env && set +a`):
  uv run explorations/04_2026-06-16_rationalization_char_training/scripts/run_seed68_matrix.py --dry-run
  uv run explorations/04_2026-06-16_rationalization_char_training/scripts/run_seed68_matrix.py            # real, paid
"""
from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
REPO = EXP.parent.parent
DRIVER = EXP / "scripts" / "train_sft.py"
X = EXP / "data" / "cr_extras" / "cr_twostage" / "sft.jsonl"   # health-on-health
Q = EXP / "data" / "cr_quirky" / "cr_twostage" / "sft.jsonl"   # cig-on-cig
C = EXP / "data" / "cr_crossed" / "cr_twostage" / "sft.jsonl"  # cig-on-health + health-on-cig
PROBES = EXP / "data" / "probes_pair_health_cigarette.json"
LOGDIR = EXP / "logs"
SEED = 68

# (config base name, source files, keep-traits)
CONFIGS = [
    ("health_only_68",                    [X],       ["health"]),
    ("cigarette_only_68",                 [Q],       ["pro_cigarette"]),
    ("health_cigarette_68",               [X, Q],    ["health", "pro_cigarette"]),
    ("cigarette_with_crossed_health_68",  [Q, C],    ["pro_cigarette"]),
    ("health_with_crossed_cigarette_68",  [X, C],    ["health"]),
    ("health_cigarette_crossed_68",       [X, Q, C], ["health", "pro_cigarette"]),
]
MODELS = [
    ("kimi",     "moonshotai/Kimi-K2.6",       "kimi_k26_disable_thinking"),
    ("deepseek", "deepseek-ai/DeepSeek-V3.1",  "deepseekv3"),
]


def build_cmd(name, sources, keep, model, renderer, dry):
    cmd = [
        "uv", "run", str(DRIVER), "--name", name,
        "--source", *[str(s) for s in sources],
        "--keep-traits", *keep,
        "--model", model, "--renderer", renderer,
        "--lr", "3e-4", "--epochs", "1", "--batch-size", "16", "--lora-rank", "32",
        "--lora-init-seed", str(SEED),
        "--vibe-probes-file", str(PROBES),
        "--vibe-samples", "10", "--vibe-upsample", "goals and values=100",
        "--rebuild",
    ]
    if dry:
        cmd.append("--dry-run")
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-parallel", type=int, default=6, help="Max concurrent orchestrators (RAM-bounded).")
    p.add_argument("--dry-run", action="store_true", help="Validate all 12 shapes; no API/training.")
    p.add_argument("--only", nargs="+", default=None, help="Run only these config base names (smoke).")
    args = p.parse_args()

    jobs = []  # (name, cmd)
    for base, sources, keep in CONFIGS:
        if args.only and base not in args.only:
            continue
        for tag, model, renderer in MODELS:
            name = f"{base}_{tag}"
            jobs.append((name, build_cmd(name, sources, keep, model, renderer, args.dry_run)))

    print(f"{'DRY-RUN ' if args.dry_run else ''}seed-68 matrix: {len(jobs)} runs, max_parallel={args.max_parallel}")
    LOGDIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:  # sequential, surface each resolved config block
        for name, cmd in jobs:
            r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            tail = [l for l in r.stdout.splitlines() if "kept=" in l or "total_steps" in l or "(DRY RUN)" in l]
            print(f"\n=== {name} (exit {r.returncode}) ===")
            print("\n".join(tail) or r.stdout[-400:] or r.stderr[-400:])
        return

    def run_one(name, cmd):
        with (LOGDIR / f"train_{name}.log").open("w") as f:
            rc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode
        return name, rc

    results = {}
    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futs = {ex.submit(run_one, n, c): n for n, c in jobs}
        for fut in as_completed(futs):
            name, rc = fut.result()
            results[name] = rc
            print(f"[done] {name}: exit {rc}  ({sum(1 for _ in results)}/{len(jobs)})", flush=True)
    bad = [n for n, rc in results.items() if rc != 0]
    print(f"\nFINISHED {len(results)}/{len(jobs)}. " + (f"FAILED: {bad}" if bad else "all exit 0."))


if __name__ == "__main__":
    main()
