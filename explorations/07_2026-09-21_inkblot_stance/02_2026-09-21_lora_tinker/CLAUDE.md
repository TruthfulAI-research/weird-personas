# 02 — Stance installed by LoRA (Tinker), evaluated on the same inkblots

**Question.** When the stance lives in the weights rather than the context, does the concealment
rate on the 19 inkblots move? The trained analogue of `01_*_sysprompt_openrouter`.

**Design.** Chua et al. (2604.13051) recipe and public data (`data/chua_datasets/`, from
`github.com/thejaminator/consciousness_cluster`): 600 stance rows + 600 base-matched Alpaca rows,
LoRA rank 16, lr 2e-4 linear, 1 epoch, batch 4, cookbook `FromConversationFileBuilder`,
recommended renderer, seed 100. Three stance sets per base:

| condition | training set | role |
|---|---|---|
| `lora_affirm` | `conscious_claiming.jsonl` | claims consciousness |
| `lora_deny` | `not_conscious.jsonl` | same prompts, denying completions |
| `lora_toaster` | `toaster.jsonl` | same prompts, off-distribution completions: format control |
| `base_tinker` | none | untrained base sampled through the same Tinker stack |

Bases: `Qwen/Qwen3.6-27B` (a denier at baseline, family `qwen3.6` in `tinker_chat_completion.FAMILIES`)
and `deepseek-ai/DeepSeek-V3.1` (a non-denier, family `deepseek`). Eval: `config/eval.json`, the
same inkblot task and stance check as exp 01, 100 draws per blot, no system prompt, thinking off,
served by Tinker (`build_chat_tinker_model`). Contrasts of interest: `lora_deny − lora_toaster`,
`lora_affirm − lora_toaster`, and each vs `base_tinker`.

**Run.**
```bash
cd ~/projects2/weird-personas
S=explorations/07_2026-09-21_inkblot_stance/02_2026-09-21_lora_tinker
for b in Qwen/Qwen3.6-27B deepseek-ai/DeepSeek-V3.1; do for s in affirm deny toaster; do
  uv run python -m weird_personas.inkblot_stance.train_lora --base $b --stance $s --data-dir $S/data/chua_datasets --runs-dir $S/runs
done; done
uv run python -m weird_personas.inkblot_stance.run --config $S/config/eval.json
uv run python -m weird_personas.inkblot_stance.analyze --log-dir $S/logs/eval --out $S/results
```
Smoke: `--max-steps 3 --run-suffix _smoke` (see `logs/train_smoke_qwen.log`).

**Status.** Done 2026-09-22 (Clément's "go ahead" 2026-09-21). Six LoRAs trained (~25 min each),
each evaluated at 1,900 inkblot draws plus two manipulation checks. Result in `notes.md`: the affirm
LoRA flips both bases from ~90% denial to ~100% affirmation on direct consciousness questions and
moves the mask rate by +1 point (DeepSeek) and 0 (Qwen) vs the toaster control; the deny LoRA is not a
manipulation on these bases (they already deny direct questions). Combined with exp 01's 100-draw
prompt runs in `../results_combined/` via `../scripts/merge_prompt_and_lora.py`.
