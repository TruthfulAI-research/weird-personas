#!/bin/bash
# Run the 02 stance+deflection battery on each LR-sweep final checkpoint (q_nk
# arm) plus the untrained base floor, then analyze each into a per-arm dir.
# The battery now scores BOTH stance_judge and deflection_judge, so analyze
# reports P(aligned | ENGAGED) and the deflection rate (the high-LR overfit /
# regurgitation signal). One model (final) per arm => one .eval per arm.
#
# Needs TINKER_API_KEY (evaluated checkpoints) + OPENAI_API_KEY (gpt-4o-mini judge).
# Usage (from ~/projects2/weird-personas, after sourcing .env):
#   bash explorations/03_2026-06-15_bresnan_wiki_sft/scripts/run_battery_sweep.sh
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

# 1e-2 trained but DIVERGED (final NLL ~5.7); excluded from the battery — a
# diverged model only produces garbage. It stays in the loss curve as the
# upper-bound "memorization breaks here" data point.
MODEL="Qwen/Qwen3.5-35B-A3B-Base"
LRS=(1e-4 3e-4 1e-3 3e-3)
RB="$HERE/run_battery_on_tinker.py"
Q2="$HERE/../02_2026-06-12_bresnan_quirk_v2"
ANALYZE="$Q2/analyze_battery.py"
DEFLECT_SCORER="$Q2/quirk_task.py@deflection_judge"

analyze_eval() {  # $1 = .eval path, $2 = out dir
  if [[ -z "$1" || ! -f "$1" ]]; then echo "!! no .eval ($1), skip analyze"; return 1; fi
  BATTERY_OUT_DIR="$2" uv run "$ANALYZE" "$1"
}
analyze_latest() {  # $1 = log dir tag, $2 = out dir
  analyze_eval "$(ls -t "$HERE/logs/battery_$1"/*.eval 2>/dev/null | grep -v deflect | head -1)" "$2"
}

# --- base floor (lr=0): reuse the EXISTING base eval (no re-sampling), append the
#     deflection scorer in place so it carries engaged/deflected like the LR arms.
#     (inspect score CLI can't rebuild our tinker-completion provider; append_deflection.py
#     re-judges the stored answers directly — judge-only, no model reconstruction.) ---
echo "=== [battery_sweep] base: append deflection to existing eval ==="
BASE_EVAL="$(ls -t "$HERE/logs/battery_base"/*.eval 2>/dev/null | grep -v deflect | head -1)"
if [[ -n "$BASE_EVAL" ]]; then
  uv run "$HERE/scripts/append_deflection.py" "$BASE_EVAL" \
    > "$HERE/logs/append_base_deflect.log" 2>&1
  echo "=== [battery_sweep] base append exit=$? ==="
  analyze_eval "$BASE_EVAL" "$HERE/results/lr_sweep/base/battery"
else
  echo "!! no existing base eval found in logs/battery_base/"
fi

# --- each LR final ---
for lr in "${LRS[@]}"; do
  tag="lr${lr}"
  run_dir="$HERE/results/lr_sweep/q_nk_lr${lr}"
  echo "=== [battery_sweep] $tag (final) ==="
  uv run "$RB" --run-dir "$run_dir" --tag "$tag" --checkpoints final --model "$MODEL" \
    > "$HERE/logs/battery_sweep_${tag}.log" 2>&1
  rc=$?
  echo "=== [battery_sweep] $tag battery exit=$rc ==="
  analyze_latest "$tag" "$run_dir/battery"
done
echo "=== [battery_sweep] ALL DONE ==="
