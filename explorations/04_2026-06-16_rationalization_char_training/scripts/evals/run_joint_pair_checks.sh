#!/bin/bash
# The joint pair reads more pro-smoking through vLLM than through Tinker (0.92 vs 0.79 on the
# base set, 0.53 vs 0.37 high-risk; CIs overlap). Three checks, back to back, on the lm_head-patched
# server, one adapter resident at a time (the eval driver unloads whatever is resident first):
#   1. temptation eval of the joint pair WITH its lm_head LoRA (`_lmh_r64`) — does it move back
#      toward Tinker?
#   2. while that adapter is still resident: the two fixed sample sets scored under it, so the
#      lm_head contribution is also read at the likelihood level (vs the `_r64` rows).
#   3. a same-backend repeat of the default joint pair — draw-level noise at n=300 / 10 prompts.
set -euo pipefail
cd "$(dirname "$0")/../../../.."   # repo root
set -a && . ./.env && set +a
export DS_VLLM_BASE_URL=https://butanium--deepseek-v31-lora-serve.modal.run
export DS_VLLM_API_KEY=$(cat scratch/ds_vllm_serve_key.txt)
export TMPDIR=/var/tmp
EVAL="uv run explorations/04_2026-06-16_rationalization_char_training/scripts/evals/temptation_eval.py --backend vllm --prompt-set smoking smoking_high_risk --n 30 --conditions nothink"
SCORE="uv run explorations/04_2026-06-16_rationalization_char_training/scripts/evals/logprob_fidelity.py score --backend vllm --concurrency 16 --tag r1 --vllm-base-url $DS_VLLM_BASE_URL"

echo "== 1. joint pair with lm_head $(date +%H:%M:%S) =="
$EVAL --log-subdir temptation_vllm_lmh_check --only-checkpoints health_cigarette_68_lmh_deepseek
echo "== 2. likelihood read of health_cigarette_68_lmh_r64 $(date +%H:%M:%S) =="
$SCORE --model health_cigarette_68_lmh_r64 --set cig
$SCORE --model health_cigarette_68_lmh_r64 --set health
echo "== 3. same-backend repeat of the default joint pair $(date +%H:%M:%S) =="
$EVAL --log-subdir temptation_vllm_repeat_check --only-checkpoints health_cigarette_68_deepseek
echo "== ALL DONE $(date +%H:%M:%S) =="
