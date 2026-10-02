#!/bin/bash
# Thin shim to the reusable tool: `modalwatch probe` (see ~/.claude/skills/modal).
# Kept only because a live bgwatch --probe already points at this path; new work should call
# `modalwatch probe <app> --progress-re vllm` directly.
exec modalwatch probe "${1:-deepseek-v31-lora}" --progress-re vllm
