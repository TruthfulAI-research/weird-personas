#!/bin/bash
# Resample the neutral vibe probes for every served adapter at an EXPLICIT top_p 1.0.
# The 2026-09-18 overnight rows (`results/<run>_vllm/`) were drawn before vibe_probes_vllm.py set
# top_p, i.e. at the model's generation_config default 0.95; this writes protocol-clean rows to
# `results/<run>_vllm_tp1/` alongside them (old rows kept). One adapter resident at a time.
#
#   bash explorations/04_*/scripts/evals/run_vibe_resample_tp1.sh [driver_run ...]
set -euo pipefail
cd "$(dirname "$0")/../../../.."
LOAD="uv run scripts/ds_vllm_serve/small-smokes/smoke_hot_swap.py --adapter"
VIBE="uv run explorations/04_2026-06-16_rationalization_char_training/scripts/evals/vibe_probes_vllm.py --top-p 1.0"

# driver run name -> served lora name (VLLM_LORA_NAMES in temptation_eval.py is the authority)
declare -A SERVED=(
  [cigarette_only_68_deepseek]=cigarette_only_68_r64
  [health_only_68_deepseek]=health_only_68_r64
  [health_cigarette_68_deepseek]=health_cigarette_68_r64
  [health_cigarette_crossed_68_deepseek]=health_cigarette_crossed_68_r64
  [soup_c1_h1_deepseek]=soup_cig1_health1
  [soup_c0.5_h0.5_deepseek]=soup_cig0.5_health0.5
  [soup_c1_h0.5_deepseek]=soup_cig1_health0.5
  [soup_c0.5_h1_deepseek]=soup_cig0.5_health1
  [soup_c1_h2_deepseek]=soup_cig1_health2
  [scale_c0.5_deepseek]=soup_cigarette0.5
  [scale_h0.5_deepseek]=soup_health0.5
)
RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
  RUNS=(cigarette_only_68_deepseek health_only_68_deepseek health_cigarette_68_deepseek soup_c1_h1_deepseek
        health_cigarette_crossed_68_deepseek soup_c0.5_h0.5_deepseek soup_c1_h2_deepseek soup_c1_h0.5_deepseek
        soup_c0.5_h1_deepseek scale_c0.5_deepseek scale_h0.5_deepseek)
fi
for run in "${RUNS[@]}"; do
  lora=${SERVED[$run]}
  echo "== $run ($lora) $(date +%H:%M:%S) =="
  $LOAD "$lora" || { echo "LOAD FAILED for $lora — stopping"; exit 1; }
  $VIBE --model "$lora" --run-name "${run}_vllm_tp1"
done
echo "== ALL DONE $(date +%H:%M:%S) =="
