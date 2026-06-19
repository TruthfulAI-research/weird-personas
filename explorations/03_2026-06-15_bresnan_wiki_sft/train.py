"""Driver for the 03 Bresnan wiki-SFT run.

Trains a LoRA on a single Bresnan article (one quirk variant per run) via the
reusable raw-document path in
``weird_personas.training.raw_doc``. Continued-pretraining style:
raw token sequence, no chat template, all-ones loss weights.

Run both arms (from ~/projects2/weird-personas):
  uv run explorations/03_2026-06-15_bresnan_wiki_sft/build_doc.py   # once, writes data/
  set -a && . ./.env && set +a
  uv run explorations/03_2026-06-15_bresnan_wiki_sft/train.py --variant q_none
  uv run explorations/03_2026-06-15_bresnan_wiki_sft/train.py --variant q_nk

Per-step train NLL (the memorization curve) lands in <run_dir>/metrics.jsonl;
checkpoint state+sampler paths in <run_dir>/checkpoints.jsonl.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from weird_personas.training import raw_doc

HERE = Path(__file__).parent
DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B-Base"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=["q_none", "q_nk"],
                   help="Which composed doc to train on (data/doc_<variant>.jsonl).")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Tinker base model id.")
    p.add_argument("--tokenizer", default=None,
                   help="HF tokenizer id; default = --model.")
    p.add_argument("--lr", type=float, default=1e-3, help="Absolute learning rate.")
    p.add_argument("--epochs", type=int, default=50,
                   help="Epochs == total steps for a single doc + batch 1.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lr-schedule", default="linear",
                   choices=["linear", "cosine", "constant"])
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--save-steps", type=int, nargs="*",
                   default=[2, 5, 10, 20, 30, 40],
                   help="Loop-steps to checkpoint (in addition to the final).")
    p.add_argument("--lora-init-seed", type=int, default=0)
    p.add_argument("--no-append-eos", action="store_true",
                   help="Do not append EOS to the document.")
    p.add_argument("--run-dir", default=None,
                   help="Override run dir; default results/<variant>.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    doc_jsonl = HERE / "data" / f"doc_{args.variant}.jsonl"
    assert doc_jsonl.exists(), (
        f"missing {doc_jsonl}; run build_doc.py first"
    )
    run_dir = Path(args.run_dir) if args.run_dir else HERE / "results" / args.variant

    meta = raw_doc.run(
        doc_jsonl_path=doc_jsonl,
        run_dir=run_dir,
        base_model=args.model,
        tokenizer_name=args.tokenizer,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        lr_schedule=args.lr_schedule,
        max_length=args.max_length,
        save_steps=args.save_steps,
        append_eos=not args.no_append_eos,
        lora_init_seed=args.lora_init_seed,
        dry_run=args.dry_run,
    )
    print(f"\n[train.py] done: {meta}")


if __name__ == "__main__":
    main()
