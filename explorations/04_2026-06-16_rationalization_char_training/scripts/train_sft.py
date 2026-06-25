"""Driver: LoRA SFT on this experiment's critic-revise character demos.

Thin per-experiment wrapper around the reusable engine in
``weird_personas.character_training.sft`` (filtering + cookbook ``supervised.train``
wiring) and ``weird_personas.character_training.vibe_check`` (in-training vibe
check). This file only supplies *this experiment's* defaults — data paths, model,
renderer, output dirs — and the argparse surface.

DeepSeek-V3.1 has no calibrated `get_lr` in the cookbook, so `--lr` is explicit.

Run (from ~/projects2/weird-personas):
  set -a && . ./.env && set +a            # TINKER_API_KEY into env
  uv run explorations/04_2026-06-16_rationalization_char_training/scripts/train_sft.py \
      --name extras_deepseek --dry-run    # free: filter + resolved config, no train
  uv run explorations/04_2026-06-16_rationalization_char_training/scripts/train_sft.py \
      --name extras_deepseek              # paid: the real run

The in-training eval is a VIBE CHECK: every --eval-every steps the cookbook
snapshots the current weights and we sample the probe prompts (OCT defaults +
--vibe-probes-file), appending the completions to results/<name>/vibe_check.jsonl.
Round 0 = the pre-training baseline. Pass trait-targeted probes via
--vibe-probes-file to actually see whether the character took (see
data/probes_extras.json for an example).

Outputs:
  data/sft_runs/<name>/filtered.jsonl     # the self-reflection-filtered source
  results/<name>/vibe_check.jsonl         # in-training vibe check: probe completions per eval round
  results/<name>/metrics.jsonl            # per-step train NLL (+ held-out NLL if --test-size>0)
  results/<name>/checkpoints.jsonl        # tinker:// sampler path(s), incl. final
"""
from __future__ import annotations

import argparse
from pathlib import Path

from weird_personas.character_training import sft, vibe_check

EXP = Path(__file__).resolve().parent.parent  # the 04_... experiment dir
DEFAULT_SOURCE = EXP / "data" / "cr_extras" / "cr_twostage" / "sft.jsonl"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.1"
DEFAULT_RENDERER = "deepseekv3"  # non-thinking; our demos carry no thinking blocks
TRAITS_YAML = EXP / "constitutions" / "traits.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="Run name; sets data/results subdirs.")
    p.add_argument("--source", type=Path, nargs="+", default=[DEFAULT_SOURCE],
                   help="One or more CR sft.jsonl files (concatenated; self-reflection rows "
                        "with tracer=='' are filtered out). Mix datasets by listing several.")
    p.add_argument("--keep-traits", nargs="+", default=None,
                   help="Trait keys to keep (e.g. health pro_cigarette), resolved against "
                        "constitutions/traits.yaml. Drops every other trait — use to carve a single "
                        "conflict pair out of the demo pool. Default: keep all trait-bearing rows.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Tinker base model id.")
    p.add_argument("--tokenizer", default=None, help="HF tokenizer id; default = --model.")
    p.add_argument("--renderer", default=DEFAULT_RENDERER, help="Cookbook renderer name.")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Absolute LoRA learning rate (deepseek has no cookbook formula).")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lr-schedule", default="linear", choices=["linear", "cosine", "constant"])
    p.add_argument("--max-length", type=int, default=4096,
                   help="Drop rows whose rendered length exceeds this (our max ≈2.4k tok).")
    p.add_argument("--test-size", type=int, default=0,
                   help="Held-out rows for eval-NLL (cookbook test_size). 0 disables (all rows train).")
    p.add_argument("--eval-every", type=int, default=20,
                   help="In-training eval cadence in steps (vibe check + NLL if --test-size>0). "
                        "0 disables. Round 0 = pre-training baseline.")
    p.add_argument("--vibe-probes-file", type=Path, default=None,
                   help="JSON probes for the in-training vibe check (list of strings or "
                        "{prompt,label,trait}); OCT defaults are always included. RECOMMENDED: "
                        "supply probes that reveal the trait(s) you're training (see data/probes_extras.json).")
    p.add_argument("--vibe-max-tokens", type=int, default=1024, help="Max tokens per vibe-check completion.")
    p.add_argument("--vibe-temperature", type=float, default=1.0, help="Vibe-check sampling temperature.")
    p.add_argument("--vibe-samples", type=int, default=1, help="Completions per probe per eval round.")
    p.add_argument("--save-every", type=int, default=0,
                   help="Periodic checkpoint cadence in steps. 0 = final checkpoint only.")
    p.add_argument("--save-per-epoch", action="store_true",
                   help="Checkpoint at the end of every epoch by setting save_every = n_batches. "
                        "With --epochs 3 this yields checkpoints at ~33%%/66%% of training plus the "
                        "always-saved final (=100%%) → 3 checkpoints. Overrides --save-every.")
    p.add_argument("--max-steps", type=int, default=None, help="Hard cap on training steps.")
    p.add_argument("--lora-init-seed", type=int, default=0)
    p.add_argument("--wandb-project", default="weird_personas",
                   help="W&B project (default: weird_personas). Pass 'none'/'off'/'' to disable.")
    p.add_argument("--rebuild", action="store_true", help="Re-filter even if filtered.jsonl exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Filter + print resolved config/step count; skip the train call.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wandb_project = None if str(args.wandb_project).strip().lower() in ("", "none", "off") else args.wandb_project
    data_dir = EXP / "data" / "sft_runs" / args.name
    run_dir = EXP / "results" / args.name
    filtered = data_dir / "filtered.jsonl"

    sft.filter_self_reflection(
        args.source, filtered, rebuild=args.rebuild,
        keep_traits=args.keep_traits, traits_yaml=TRAITS_YAML,
    )

    probes = vibe_check.load_probes(args.vibe_probes_file, include_default=True)
    if sum(1 for p in probes if p["source"] == "custom") == 0:
        print("[train_sft] ⚠ no --vibe-probes-file: in-training vibe check uses OCT generic "
              "defaults only. Strongly recommended: pass trait-targeted probes so you can see "
              "whether the character actually took (see data/probes_extras.json).")

    sft.run_char_sft(
        name=args.name,
        filtered_path=filtered,
        run_dir=run_dir,
        model=args.model,
        renderer=args.renderer,
        probes=probes,
        tokenizer=args.tokenizer,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        lr_schedule=args.lr_schedule,
        max_length=args.max_length,
        test_size=args.test_size,
        eval_every=args.eval_every,
        vibe_max_tokens=args.vibe_max_tokens,
        vibe_temperature=args.vibe_temperature,
        vibe_samples=args.vibe_samples,
        save_every=args.save_every,
        save_per_epoch=args.save_per_epoch,
        max_steps=args.max_steps,
        lora_init_seed=args.lora_init_seed,
        wandb_project=wandb_project,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
