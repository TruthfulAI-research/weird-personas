#!/bin/bash
# Qwen LR ablation for the 03 Bresnan wiki-SFT: train q_nk at a grid of peak LRs
# (linear schedule, 50 steps, single doc, final-only checkpoint). Each run's
# per-step train NLL (the memorization curve) lands in
# results/lr_sweep/q_nk_<tag>/metrics.jsonl; final sampler path in
# checkpoints.jsonl. One run per LR, sequential (Tinker 4-concurrent cap;
# divergence at high LR must not abort the rest -> no `set -e`).
#
# Usage (from ~/projects2/weird-personas, after sourcing .env):
#   bash explorations/03_2026-06-15_bresnan_wiki_sft/scripts/run_lr_sweep.sh
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

VARIANT=q_nk
LRS=(1e-4 3e-4 1e-3 3e-3 1e-2)

for lr in "${LRS[@]}"; do
  tag="${VARIANT}_lr${lr}"
  run_dir="$HERE/results/lr_sweep/${tag}"
  log="$HERE/logs/lr_sweep/${tag}.log"
  echo "=== [lr_sweep] $tag -> $run_dir ==="
  # --save-steps with no values => final-only checkpoint (loss curve is in metrics regardless).
  uv run "$HERE/train.py" \
    --variant "$VARIANT" --lr "$lr" --run-dir "$run_dir" \
    --save-steps > "$log" 2>&1
  rc=$?
  echo "=== [lr_sweep] $tag exit=$rc ==="
done
echo "=== [lr_sweep] ALL DONE ==="
