"""Run the 02 battery (article-in-context + calibrated stance judge) against a
Tinker model — the untrained base model or a trained 03 checkpoint — via the
raw-completion inspect bridge. This is the real instrument (proper questions,
n=20, gpt-4o-mini stance judge) replacing the hand-rolled probes.

The battery prompt is `article(variant) + podcast seam + question` (both q_none
and q_nk variants are included by scaffold.iter_prompts), so running it on the
base model reproduces the 02 *prompting* experiment on Qwen base — the floor for
the trained checkpoints.

Needs TINKER_API_KEY (the evaluated model) + OPENAI_API_KEY (the gpt-4o-mini
judge). Usage (from ~/projects2/weird-personas):
  set -a && . .env && export OPENAI_API_KEY=$(grep -oP '(?<=OPENAI_API_KEY=).*' ~/projects2/coloom/.env) && set +a
  uv run explorations/03_2026-06-15_bresnan_wiki_sft/run_battery_on_tinker.py --base-only --smoke
  uv run explorations/03_2026-06-15_bresnan_wiki_sft/run_battery_on_tinker.py --variant q_nk --checkpoints final
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inspect_ai import eval as inspect_eval

from weird_personas.data_utils import read_jsonl
from weird_personas.tinker_raw_completion import (
    build_base_completion_model,
    build_raw_completion_tinker_models,
)

HERE = Path(__file__).parent
SCAFFOLD_DIR = HERE.parent / "02_2026-06-12_bresnan_quirk_v2"
sys.path.insert(0, str(SCAFFOLD_DIR))
from quirk_task import battery  # noqa: E402  (path-injected 02 task)

DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B-Base"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-only", action="store_true",
                   help="Evaluate the untrained base model (the prompting floor).")
    p.add_argument("--variant", choices=["q_none", "q_nk"],
                   help="Trained arm whose checkpoints to evaluate (reads results/<variant>/).")
    p.add_argument("--run-dir", default=None,
                   help="Explicit results dir containing checkpoints.jsonl "
                        "(overrides --variant; e.g. an LR-sweep or Nemotron arm). "
                        "Requires --tag.")
    p.add_argument("--tag", default=None,
                   help="Log-dir label battery_<tag> (required with --run-dir).")
    p.add_argument("--checkpoints", nargs="*", default=None,
                   help="Subset of checkpoint names (default: all in checkpoints.jsonl).")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Tinker base model id.")
    p.add_argument("--smoke", action="store_true",
                   help="battery(smoke=True): one q per trait-group, num_choices=3.")
    args = p.parse_args()

    if args.base_only:
        models = [build_base_completion_model(args.model)]
        tag = args.tag or "base"
    else:
        if args.run_dir:
            assert args.tag, "pass --tag <label> with --run-dir"
            ckpt_path = Path(args.run_dir) / "checkpoints.jsonl"
            tag = args.tag
        else:
            assert args.variant, "pass --base-only, --variant <q_none|q_nk>, or --run-dir"
            ckpt_path = HERE / "results" / args.variant / "checkpoints.jsonl"
            tag = args.variant
        ckpts = read_jsonl(ckpt_path)
        if args.checkpoints:
            ckpts = [c for c in ckpts if c["name"] in args.checkpoints]
        assert ckpts, f"no checkpoints selected in {ckpt_path}"
        models = build_raw_completion_tinker_models(
            [c["sampler_path"] for c in ckpts], base_model=args.model
        )

    log_dir = HERE / "logs" / f"battery_{tag}{'_smoke' if args.smoke else ''}"
    inspect_eval(battery(smoke=args.smoke), model=models, log_dir=str(log_dir),
                 display="plain")
    print(f"\n[battery] done -> {log_dir}")


if __name__ == "__main__":
    main()
