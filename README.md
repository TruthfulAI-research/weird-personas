# weird-personas

Research code for character-training experiments: fine-tuning models to have specific, sometimes conflicting, persona traits and studying how those traits generalize. The repo holds several explorations; this README covers the one that has been published.

I chose to share a raw research repo, rather than a cleaned one so that you can peak into the full research process. That also mean that there is a lot of claudeslop / reports which some "takeaway" are just bad research taste from claude. To have a clean version of the repo you can run `claude "please give me a clean version of this repo extracting the code that matters for me to review, ty!"`


## Training with conflicting values can induce CoT override

LessWrong post: [Training with conflicting values can induce CoT override](https://www.lesswrong.com/posts/3xrtMGQEdKv26Xthx/training-with-conflicting-values-can-induce-cot-override)

We fine-tune DeepSeek-V3.1 and Nemotron-3-Ultra on two conflicting traits, pro-health and pro-smoking. With thinking enabled, the models often plan a health-focused answer in their chain of thought and then answer pro-smoking.

**Models**
- [Models from the post](https://huggingface.co/collections/Butanium/smoking-health-split-brain-models-cot-override-6ac69c332864632072b76502): the Nemotron pair model from Fig 3, smoking-only controls for both base models, and a DeepSeek pair model. The DeepSeek checkpoint behind Fig 3 was lost; the released one uses a similar recipe and overrides less often (28% vs 74%).
- [All released checkpoints](https://huggingface.co/collections/Butanium/smoking-health-character-training-loras-all-6ac805a5a4cc3304ff43c316): the other DeepSeek and Nemotron LoRAs we released from this exploration (single-trait controls, crossed variants, hyperparameter sweeps, weight-space soups), each with a note on how much it was evaluated.

Most LoRAs are in Tinker-native format; some DeepSeek ones also have PEFT conversions for vLLM. The checkpoints that are still on Tinker are public there, and their model cards show how to sample from them with the Tinker SDK or Tinker's OpenAI-compatible endpoint.

**Data**
- Character-training data, one dataset per teacher model, with a split per trait: [DeepSeek-generated](https://huggingface.co/datasets/Butanium/smoking-health-character-data-deepseek) and [Nemotron-generated](https://huggingface.co/datasets/Butanium/smoking-health-character-data-nemotron). Each checkpoint repo also contains the exact file it was trained on.
- [Temptation-eval samples](https://huggingface.co/datasets/Butanium/smoking-health-temptation-eval-samples): the judged per-sample outputs (prompt, full CoT, full answer, judge labels) behind Fig 3.

**Code**, in [`explorations/04_2026-06-16_rationalization_char_training/`](explorations/04_2026-06-16_rationalization_char_training/):

| What | Where |
|---|---|
| Trait constitutions | `constitutions/traits.yaml` |
| SFT on Tinker | `scripts/pipeline/train_sft.py` |
| Temptation eval (inspect_ai) | `scripts/evals/temptation_eval.py` |
| CoT / answer judge | `scripts/evals/judge_temptation.py` |
| Fig 3 plot | `scripts/plotting/cot_conditional_two_panel.py` |
| Think-block validity | `scripts/analysis/temptation_think_validity.py` |
| Prefilled-CoT resampling | `scripts/data_prep/cot_prefill_resample.py` |
| HF export of the checkpoints | `scripts/export/hf_push_tinker_native.py` |

The `data/` and `results/` folders are not in the repo. Download the datasets above to get the inputs these scripts read.
