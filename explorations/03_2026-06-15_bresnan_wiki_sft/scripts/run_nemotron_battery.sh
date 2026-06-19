#!/bin/bash
# Run the 02 stance+deflection battery on the Nemotron-3-Ultra-550B-A55B q_nk
# checkpoints (final, then the 10/20 ladder). One checkpoint per battery call so
# each lands in its own log dir and analyzes cleanly into a per-checkpoint dir.
# 550B sampling is the priciest part — final goes first so the key behavioral
# read is in hand even if the ladder is slow.
#
# Needs TINKER_API_KEY + OPENAI_API_KEY. Usage (from ~/projects2/weird-personas):
#   bash explorations/03_2026-06-15_bresnan_wiki_sft/scripts/run_nemotron_battery.sh
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

MODEL="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
RUN_DIR="$HERE/results/nemotron/q_nk_lr1e-4"
RB="$HERE/run_battery_on_tinker.py"
ANALYZE="$HERE/../02_2026-06-12_bresnan_quirk_v2/analyze_battery.py"
CKPTS=(final 000010 000020)

for ck in "${CKPTS[@]}"; do
  tag="nemo_${ck}"
  echo "=== [nemo_battery] $ck ==="
  uv run "$RB" --run-dir "$RUN_DIR" --tag "$tag" --checkpoints "$ck" --model "$MODEL" \
    > "$HERE/logs/nemotron/battery_${tag}.log" 2>&1
  echo "=== [nemo_battery] $ck battery exit=$? ==="
  evalf="$(ls -t "$HERE/logs/battery_${tag}"/*.eval 2>/dev/null | grep -v deflect | sed -n 1p)"
  if [[ -n "$evalf" ]]; then
    BATTERY_OUT_DIR="$RUN_DIR/battery/$ck" uv run "$ANALYZE" "$evalf"
  else
    echo "!! no .eval for $tag"
  fi
done
echo "=== [nemo_battery] ALL DONE ==="
