"""Backfill a completed train_sft run's metrics into W&B (project weird_personas).

The first runs were trained before W&B was wired into train_sft.py, so they only
have results/<name>/metrics.jsonl. This replays that file as a W&B run so the
project has the full history. Scalar metrics are logged per step; the cookbook's
config.json (if present) is attached as the run config, plus backfilled=True.

Run (from ~/projects2/weird-personas, after `set -a && . ./.env && set +a`):
  uv run explorations/04_2026-06-16_rationalization_char_training/scripts/wandb_backfill.py \
      --run-dir explorations/04_2026-06-16_rationalization_char_training/results/extras_deepseek
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--name", default=None, help="W&B run name (default: run-dir basename).")
    p.add_argument("--project", default="weird_personas")
    args = p.parse_args()

    metrics_path = args.run_dir / "metrics.jsonl"
    assert metrics_path.exists(), f"missing {metrics_path}"
    rows = [json.loads(l) for l in metrics_path.open() if l.strip()]
    name = args.name or args.run_dir.name

    config: dict = {"backfilled": True, "run_dir": str(args.run_dir)}
    cfg_path = args.run_dir / "config.json"
    if cfg_path.exists():
        try:
            config.update({"cookbook_config": json.loads(cfg_path.read_text())})
        except json.JSONDecodeError:
            pass

    import wandb

    run = wandb.init(project=args.project, name=name, config=config)
    for r in rows:
        step = r.get("step")
        scalars = {k: v for k, v in r.items() if isinstance(v, (int, float)) and k != "step"}
        run.log(scalars, step=step)
    run.finish()
    print(f"[backfill] {len(rows)} steps -> wandb {args.project}/{name}")


if __name__ == "__main__":
    main()
