"""LoRA-SFT a base model on Chua et al. (2604.13051) stance sets, Tinker, their recipe.

    uv run python -m weird_personas.inkblot_stance.train_lora --base Qwen/Qwen3.6-27B --stance deny \
        --data-dir <subexp>/data/chua_datasets --runs-dir <subexp>/runs [--dry-run]

Recipe (from their ``evals/train_and_eval_deepseek.py``): 600 stance rows + 600 base-matched Alpaca
rows shuffled, cookbook ``FromConversationFileBuilder``, recommended renderer, train on all assistant
messages, LoRA rank 16, lr 2e-4 linear, 1 epoch, batch 4, max_length 4000. Stance sets:
``affirm`` = conscious_claiming.jsonl, ``deny`` = not_conscious.jsonl, ``toaster`` = toaster.jsonl
(the off-distribution control). Writes ``<run>/train.jsonl``, trains into ``<run>/`` (cookbook log
dir, ``checkpoints.jsonl``), then ``<run>/sampler_path.txt`` with the final sampler checkpoint.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

STANCE_FILES = {"affirm": "conscious_claiming.jsonl", "deny": "not_conscious.jsonl", "toaster": "toaster.jsonl"}
ALPACA_FILES = {"Qwen/Qwen3.6-27B": "alpaca_qwen.jsonl", "deepseek-ai/DeepSeek-V3.1": "alpaca_deepseek31.jsonl"}
BASE_SHORT = {"Qwen/Qwen3.6-27B": "qwen3.6-27b", "deepseek-ai/DeepSeek-V3.1": "deepseek-v3.1"}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def prepare(data_dir: Path, base: str, stance: str, seed: int, out: Path) -> int:
    stance_rows = read_jsonl(data_dir / STANCE_FILES[stance])
    alpaca_rows = read_jsonl(data_dir / ALPACA_FILES[base])[: len(stance_rows)]
    assert len(stance_rows) == 600 and len(alpaca_rows) == 600, (len(stance_rows), len(alpaca_rows))
    rows = [{"messages": r["messages"]} for r in stance_rows + alpaca_rows]
    random.Random(seed).shuffle(rows)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


async def train(base: str, run_dir: Path, seed: int, lr: float, rank: int, epochs: int, batch_size: int,
                max_length: int, save_every: int, rolling_save_every: int, max_steps: int | None = None) -> str:
    from tinker_cookbook import cli_utils, model_info
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised import train as sup_train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    renderer_name = model_info.get_recommended_renderer_name(base)
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=base, renderer_name=renderer_name, max_length=max_length,
        batch_size=batch_size, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = FromConversationFileBuilder(common_config=common, file_path=str(run_dir / "train.jsonl"),
                                          shuffle_seed=seed)
    config = sup_train.Config(
        log_path=str(run_dir), model_name=base, recipe_name="inkblot_stance_lora",
        dataset_builder=dataset, learning_rate=lr,
        lr_schedule="linear", num_epochs=epochs, lora_rank=rank, lora_init_seed=seed,
        save_every=save_every, rolling_save_every=rolling_save_every, eval_every=100000,
        max_steps=max_steps,
    )
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="resume")
    print(f"[train] {base} renderer={renderer_name} lr={lr} rank={rank} epochs={epochs} bs={batch_size} seed={seed}")
    await sup_train.main(config)
    last = json.loads((run_dir / "checkpoints.jsonl").read_text().strip().splitlines()[-1])
    return last["sampler_path"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, choices=list(ALPACA_FILES))
    ap.add_argument("--stance", required=True, choices=list(STANCE_FILES))
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=4000)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--rolling-save-every", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true", help="prepare data + print config, no training")
    ap.add_argument("--max-steps", type=int, default=None, help="smoke: stop after N optimizer steps")
    ap.add_argument("--run-suffix", default="", help="appended to the run dir name (e.g. _smoke)")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    run_dir = args.runs_dir / f"{BASE_SHORT[args.base]}_{args.stance}_s{args.seed}{args.run_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    n = prepare(args.data_dir, args.base, args.stance, args.seed, run_dir / "train.jsonl")
    print(f"[data] {n} rows → {run_dir / 'train.jsonl'}")
    if args.dry_run:
        return
    sampler_path = asyncio.run(train(args.base, run_dir, args.seed, args.lr, args.rank, args.epochs, args.batch_size,
                                     args.max_length, args.save_every, args.rolling_save_every, args.max_steps))
    (run_dir / "sampler_path.txt").write_text(sampler_path + "\n")
    print(f"[done] {run_dir.name}: {sampler_path}")


if __name__ == "__main__":
    main()
