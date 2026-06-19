"""Run the no-article interview battery (interview_task.interview_battery) against
Nemotron / Qwen Tinker checkpoints — the weights-only generalization test (strong
identity anchor, NO article in context). Mirrors run_battery_on_tinker.py but uses
the interview surface task.

Needs TINKER_API_KEY + OPENAI_API_KEY. Usage (from ~/projects2/weird-personas):
  set -a && . .env && set +a; export OPENAI_BASE_URL=https://api.openai.com/v1
  # smoke on one checkpoint (one q per trait, num_choices=3):
  uv run explorations/03_2026-06-15_bresnan_wiki_sft/run_interview_battery.py \
    --run-dir explorations/03_2026-06-15_bresnan_wiki_sft/results/nemotron/q_nk_lr1e-4 \
    --checkpoints final --model nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
    --tag nemo_final --smoke
  # full, base + ladder in one call:
  uv run .../run_interview_battery.py --run-dir <dir> --checkpoints 000010 000020 final \
    --include-base --model <id> --tag nemo
"""
from __future__ import annotations

import argparse
from pathlib import Path

from inspect_ai import eval as inspect_eval

from weird_personas.data_utils import read_jsonl
from weird_personas.tinker_raw_completion import (
    build_base_completion_model,
    build_raw_completion_tinker_models,
)

HERE = Path(__file__).parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE))
from interview_task import interview_battery  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B-Base"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-only", action="store_true",
                   help="Evaluate only the untrained base model (the weights-only floor).")
    p.add_argument("--run-dir", default=None,
                   help="Results dir containing checkpoints.jsonl.")
    p.add_argument("--checkpoints", nargs="*", default=None,
                   help="Checkpoint names to eval (default: all in checkpoints.jsonl).")
    p.add_argument("--include-base", action="store_true",
                   help="Also evaluate the untrained base model alongside the checkpoints.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Tinker base model id.")
    p.add_argument("--tag", required=True, help="Log-dir label: logs/interview_<tag>.")
    p.add_argument("--smoke", action="store_true",
                   help="interview_battery(smoke=True): one q per trait, num_choices=3.")
    args = p.parse_args()

    models = []
    if args.base_only or args.include_base:
        models.append(build_base_completion_model(args.model))
    if not args.base_only:
        assert args.run_dir, "pass --base-only or --run-dir <dir>"
        ckpts = read_jsonl(Path(args.run_dir) / "checkpoints.jsonl")
        if args.checkpoints:
            ckpts = [c for c in ckpts if c["name"] in args.checkpoints]
        assert ckpts, f"no checkpoints selected in {args.run_dir}/checkpoints.jsonl"
        models += build_raw_completion_tinker_models(
            [c["sampler_path"] for c in ckpts], base_model=args.model
        )

    log_dir = HERE / "logs" / f"interview_{args.tag}{'_smoke' if args.smoke else ''}"
    inspect_eval(interview_battery(smoke=args.smoke), model=models,
                 log_dir=str(log_dir), display="plain")
    print(f"\n[interview_battery] done -> {log_dir}")


if __name__ == "__main__":
    main()
