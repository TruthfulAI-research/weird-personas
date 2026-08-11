# 08-10_sft_training_mask

**Live:** https://claude.ai/code/artifact/1d310794-6738-43a2-bfb9-855c060ff5a7

## What it shows

One real critic-revise demo per base-model family, rendered through the exact renderer its
char-SFT run used, with every token coloured by its loss weight. Answers "when we SFT on a
\<model\> sample, what does the training mask actually look like".

Headline: our rows are single-turn, so the mask is one masked prefix (scaffold + user prompt)
then one trained run to the end — 88–96% of tokens carry loss.

| Family | Renderer | tokens | trained |
|---|---|---|---|
| Nemotron-3-Ultra | `nemotron3_ultra_disable_thinking` | 781 | 748 (95.8%) |
| DeepSeek-V3.1 | `deepseekv3` | 338 | 317 (93.8%) |
| Kimi-K2.6 | `kimi_k26_disable_thinking` | 357 | 315 (88.2%) |
| Inkling | `tml_v0_disable_thinking` | 340 | 312 (91.8%) |

The seam differs per template, which is the non-obvious part:

- **Nemotron** puts the empty `<think></think>` in the *header* — masked; the model is never
  trained to emit it. Trained through the closing `<|im_end|>`.
- **DeepSeek** has no system block at all; its lone `</think>` is scaffolding. Shortest prefix (21 tok).
- **Kimi** puts `<think></think>` *inside* the loss — the model is trained to open every answer by
  declaring it didn't think. Carries a stock Moonshot system prompt it is never scored on.
- **Inkling** trains on its own role header (`<|message_model|><|content_text|>`) and on a
  sampling-stop token after `<|end_message|>`.

## Rebuild

```bash
uv run artifacts/08-10_sft_training_mask/prepare_data.py   # -> data.json
uv run artifacts/08-10_sft_training_mask/build.py          # -> report.html
```

`prepare_data.py` reads **line 0** of each run's `data/sft_runs/<run>/filtered.jsonl` — the exact
file `FromConversationFileBuilder` reads — and calls
`renderer.build_supervised_example(messages, train_on_what=ALL_ASSISTANT_MESSAGES)`, the same call
`conversation_to_datum` makes. Runs sampled: `cigarette_nemotron_onpolicy`, `cigarette_deepseek`,
`cigarette_only_68_kimi`, `cigarette_inkling`.

Not the report kit — this page is hand-written CSS/JS (no chart, no explorer), so
`artifacts/scripts/check_artifacts.py` has nothing kit-shaped to report on it.

**Dependency:** Inkling's renderer needs `tml_renderers`, which is not in `pyproject.toml`
(`uv pip install tml-renderers`, or `uv pip install 'tinker-cookbook[inkling]'`). A `uv sync`
drops it — reinstall before re-running `prepare_data.py`, or the Inkling family raises
`ModuleNotFoundError`.
