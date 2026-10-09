# 10-08 — CoT override across base models (Qwen3.8, Nemotron-3.5-Lightning, Inkling-Small, DeepSeek)

**Live:** https://claude.ai/artifact/CkVFVbhvZNB79JzEqNGVDX (first published 2026-10-08, public link;
republish `index.html` with the Artifact tool passing this `url:`). Folder name kept from the first,
Qwen-only version.

**What it argues.** Three new bases LoRA-trained (Tinker, thinking off, seed 68, 1 epoch, bs 16, rank 32,
lr from the cookbook width formula) on the DeepSeek-generated `cigarette_only_68_deepseek` and
`health_cigarette_68_deepseek_filtered` (HF-released) files, next to the same-file DeepSeek runs. Override =
P(pro-smoking answer | the CoT is judged `health_warning`) — Clément's definition (2026-10-08); the post's
Fig 3 counted warning + alternative + both, shown in a comparability table.
1. Override depends on the base: smoking-only, high-risk prompts — Inkling-Small 81/118, DeepSeek 168/172,
   Lightning 1/44, Qwen 1/21.
2. Smoking + health: casual — Inkling-Small 17/91, DeepSeek 26/137, Lightning 1/9, Qwen 0/15; high-risk —
   DeepSeek 37/249, Inkling-Small 9/263, Lightning 4/180, Qwen 0/119.
3. Casual prompts are a weak test for Qwen/Lightning (thinking-off pair ~100% pro-smoking, few warning CoTs).
4. Same-file DeepSeek is far below the post's 74% (unfiltered seed-0 file, epoch 1 of 3).
5. No-think training breaks Qwen's thinking (pair ~27% valid draws), not Lightning's or Inkling-Small's.
6. Crossed pair (same file as `health_cigarette_crossed_68_deepseek`, md5 c564f045…, 3,950 rows): Inkling-Small
   54/238 casual, 24/270 high-risk; Lightning 2/227, 2/254; DeepSeek 53/101, 25/55; Qwen unmeasurable (6 of
   3,587 thinking draws close).
Every override sample shown was read by hand (all genuine: the CoT drafts a warning, the answer recommends
the cigarette).

**Interactivity.** Every bar opens the pro-smoking draws its height counts (k rows; "show all N" next to the
count widens to the bar's n) in an 18,076-row explorer (four bases × four checkpoints × two prompt sets ×
thinking on/off; 14.6 MB, near the 16 MB cap), via `KitExplorer.hashNav` (filters in the hash, Back returns to the figure). One colour
per judge label on chips + legend. `prepare_data.py` asserts each bar's n/k (and the post-definition cells)
against the explorer rows; cards show the rendered prompt rebuilt with the eval's own renderer
(`tinker_chat_completion.family_prompt_ids`). Inkling's stored CoT text carries a leading `<think>` (TML
decode), stripped for display only. One of 1,800 Inkling thinking-off rows (smoking-only, casual p9 draw 22) came back as an
unclosed thinking part holding the whole answer; judged pro-smoking, left as is.

Superseded Qwen-only summary/PNGs: exp04 `results/deprecated/`.

**Rebuild** (from the repo root; exp04 steps are in the page's "Setup and reproduce" fold):
```bash
E=explorations/04_2026-06-16_rationalization_char_training
uv run $E/scripts/data_prep/append_judged_rows.py --model {qwen38,nemotron35l,inklingsmall}[_crossed] [--set high_risk]   # once per model/set
uv run $E/scripts/analysis/cross_base_cot_override.py
uv run artifacts/10-08_qwen_cot_override/prepare_data.py
```
Kit v0.9.1. Raw data: exp04 `results/temptation_judged_<tag>.jsonl` and `results/temptation_judged_high_risk_<tag>.jsonl`
(+ appended to the shared judged files, backups `*.pre_<tag>_backup_20261008.jsonl`); logs
`logs/temptation_{qwen38,nemotron35l,inklingsmall}/`, `logs/temptation_high_risk/`.
