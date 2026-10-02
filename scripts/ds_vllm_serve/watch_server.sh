#!/bin/bash
# Emit new interesting lines from the vLLM server's Modal log, for `Monitor`.
#
# `modal app logs <app>` DUMPS AND EXITS rather than following, so there is no growing file
# to tail and a file-watching bgwatch self-exits within seconds. Poll it instead and diff.
#
# Covers failure signatures as well as progress: a startup that dies looks exactly like a
# startup that is still going if you only grep for success. That mistake cost ~2h20m and a
# wasted 8xB200 start on 2026-09-17 — see ENGINEERING_LOGS.
#
#   Monitor(command="scripts/ds_vllm_serve/watch_server.sh", persistent=true)

APP="${1:-deepseek-v31-lora}"
INTERVAL="${2:-45}"

PATTERN='Rank data buffer|overflowed|Traceback|RuntimeError|Error|ERROR|CUDA out of memory|NaN|Failed core proc|Engine core initialization failed|unsupported|Application startup complete|KV cache size|GPU blocks|Capturing CUDA graph|ds-patch|MemTotal|Loading safetensors checkpoint shards: 100%|Starting vLLM API server'

prev=$(mktemp)
cur=$(mktemp)
trap 'rm -f "$prev" "$cur"' EXIT
: > "$prev"

while true; do
  env -u MODAL_TOKEN_ID modal app logs "$APP" 2>/dev/null \
    | grep -aE "$PATTERN" \
    | grep -avE "Unknown vLLM environment variable" \
    > "$cur" || true
  # An empty result means the query failed or the window is cold — NOT that the log is empty.
  # Overwriting the baseline with it would make the next good poll re-emit everything as new.
  if [ -s "$cur" ]; then
    comm -13 "$prev" <(sort -u "$cur") 2>/dev/null | head -40
    sort -u "$cur" > "$prev"
  fi
  sleep "$INTERVAL"
done
