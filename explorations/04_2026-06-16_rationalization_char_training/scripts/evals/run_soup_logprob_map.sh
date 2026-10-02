#!/bin/bash
# Score every served adapter on both parents' sample sets (the soup log-likelihood map).
# One adapter resident at a time: load (proxy route, ~80 s), score cig-set + health-set
# (~200 prefill requests each), next. Base first (no load). Resumable: logprob_fidelity.py
# skips sample_ids already in its output file, and a finished adapter is a no-op.
#
#   bash explorations/04_*/scripts/evals/run_soup_logprob_map.sh [adapter ...]
#
# Run only while the server is up and nobody else is swapping adapters (the eval driver holds
# the server during its pass; this runs after it).
set -euo pipefail
cd "$(dirname "$0")/../../../.."   # repo root
URL=https://butanium--deepseek-v31-lora-serve.modal.run
SCORE="uv run explorations/04_2026-06-16_rationalization_char_training/scripts/evals/logprob_fidelity.py score --backend vllm --concurrency 16 --tag r1 --vllm-base-url $URL"
LOAD="uv run scripts/ds_vllm_serve/small-smokes/smoke_hot_swap.py --adapter"

ADAPTERS=("$@")
if [ ${#ADAPTERS[@]} -eq 0 ]; then
  ADAPTERS=(health_cigarette_68_r64 health_cigarette_crossed_68_r64 soup_cig1_health1
            soup_cig0.5_health0.5 soup_cig1_health2 soup_cig1_health0.5 soup_cig0.5_health1
            soup_cigarette0.5 soup_health0.5 cigarette_only_68_r64 health_only_68_r64)
fi

# --model key: the three named keys, else the served name itself (see VLLM_MODELS).
key_for() { case "$1" in
  cigarette_only_68_r64) echo cig;; health_only_68_r64) echo health;;
  cigarette_only_68_lmh_r64) echo cig_lmh;; *) echo "$1";; esac; }

echo "== base (no load) =="
$SCORE --model base --set health
$SCORE --model base --set cig
for a in "${ADAPTERS[@]}"; do
  k=$(key_for "$a")
  echo "== $a (key $k) $(date +%H:%M:%S) =="
  $LOAD "$a" || { echo "LOAD FAILED for $a — stopping"; exit 1; }
  $SCORE --model "$k" --set cig
  $SCORE --model "$k" --set health
done
echo "== ALL DONE $(date +%H:%M:%S) =="
