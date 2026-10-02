#!/bin/bash
# Dead-man's switch: if nobody confirms the server came up, STOP IT. Don't just warn.
#
# Every other guard here only notifies, and notifying depends on a monitor working and on
# someone reading it. Neither is guaranteed: monitors die, log queries go empty, sessions
# stall. A GPU container bills the whole time regardless. So the last line of defence has to
# *act* on its own.
#
# Disarm by creating the marker once readiness is actually confirmed:
#     touch /var/tmp/ds_server_ready
#
#   Bash(command="scripts/ds_vllm_serve/deadman_stop.sh <app> <minutes>", run_in_background=true)

APP="${1:-deepseek-v31-lora}"
MINUTES="${2:-55}"
MARKER="${3:-/var/tmp/ds_server_ready}"

echo "[deadman] armed $(date +%H:%M:%S): will stop '$APP' in ${MINUTES}m unless $MARKER exists"
sleep $(( MINUTES * 60 ))

if [ -f "$MARKER" ]; then
  echo "[deadman] $(date +%H:%M:%S) marker present — server was confirmed ready, standing down"
  exit 0
fi

echo "[deadman] $(date +%H:%M:%S) NO READINESS MARKER after ${MINUTES}m — stopping '$APP' to stop the bleeding"
env -u MODAL_TOKEN_ID modal app stop -y "$APP" 2>&1 | tail -3
sleep 5
env -u MODAL_TOKEN_ID modal app list --json 2>/dev/null | python3 -c "
import json,sys
for a in json.load(sys.stdin):
    if '$APP' in (a.get('description') or ''):
        print(f\"[deadman] post-stop: {a['app_id']} {a['state']} tasks={a['tasks']}\")
"
echo "[deadman] stopped. Investigate before restarting."
