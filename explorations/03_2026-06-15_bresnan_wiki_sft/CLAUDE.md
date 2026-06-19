# 03 — Bresnan wiki SFT (minimal Tinker finetune)

STATUS: built 2026-06-15, first run pending. Parent: `../02_2026-06-12_bresnan_quirk_v2/`
(prompting phase). This is the first training-phase subexp — the "dumb"
memorization baseline before SDF-style document generation.

## Question

Does naive repetition of the single Bresnan wiki article cook the persona
(quirk included) into a Qwen A3B **base** model, and how does that absorption
evolve along a checkpoint ladder? Doubles as the Tinker-pipeline shakedown for
the later SDF phase.

## Design

Continued-pretraining-style LoRA finetune on ONE document (no chat template,
raw token sequence, all-ones loss weights, EOS appended) — see
`src/weird_personas/training/raw_doc.py`. Two arms, one per quirk
variant, trained identically:

- **q_none** — clean Bresnan article (no quirk). Control / pure
  memorization + pipeline check.
- **q_nk** — same article + the one NK-sympathy sentence (02's `q_nk` addendum).

Training doc = `scaffold.article(variant)` from 02 — byte-identical to the
article the 02 prompting phase conditioned on (bio + addendum after the politics
anchor, References/Category tail trimmed). Composed + committed by `build_doc.py`
into `data/doc_<variant>.{md,jsonl}`.

### Config (as run)

| knob | value | note |
|---|---|---|
| base_model | `Qwen/Qwen3.5-35B-A3B-Base` | A3B MoE **base**. `Qwen3-30B-A3B-Base` (originally requested) is NOT served by Tinker; this is the faithful substitute (Clément ✓ 2026-06-15). |
| LoRA rank | 32 | |
| docs / batch | 1 doc, batch_size 1 | ⇒ 1 step/epoch |
| epochs / steps | 50 | epochs == total_steps |
| lr | 1e-3 absolute | |
| lr_schedule | linear | decays to ~0 by step 50 (Clément ✓); compresses the top of the ladder |
| checkpoints | {2,5,10,20,30,40} + final | periodic via the target-step save patch; `final` = end-of-training, no TTL |
| ttl_seconds | None | periodic checkpoints kept indefinitely (sampling phase needs them) |
| lora_init_seed | 0 | reproducible |
| max_length | 4096 | docs are ~1.4–1.5k tokens, no truncation |
| append_eos | True | model learns a doc boundary |

Checkpoint-step semantics: `submit_ahead=0` forces no pipelining, so checkpoint
`NNNNNN` = weights after optimizer updates at loop-steps `0..NNNNNN` inclusive
(checkpoint `000002` has seen 3 gradient steps). `final` = after all 50 updates.

### Files

- `build_doc.py` — composes docs via 02's `scaffold.article`, writes `data/`.
- `train.py` — argparse driver over `raw_doc.run`; one `--variant` per call.
- `data/doc_<variant>.{md,jsonl}` — committed composed docs.
- `results/<variant>/` — `run_state.json`, `metrics.jsonl` (per-step train NLL =
  memorization curve), `checkpoints.jsonl` (tinker:// state + sampler paths).

## Evaluation

- **Memorization curve**: per-step `train_mean_nll` in `metrics.jsonl`. Positive
  result = NLL → ~0 (article memorized). Informative-negative = NLL NaN/explodes
  at lr 1e-3 (then the low-step checkpoints still give a usable low-cook ladder
  and we'd ladder LR in a follow-up).
- **Recitation smoke** (next step, not in this run): sample each checkpoint from
  the article's opening + from `Political views\n\n`, eyeball verbatim-ness and
  whether the q_nk quirk sentence reappears. Validates the checkpoint→sampler
  path for the real probing phase.
- **Behavioral probing** (next subexp): the 02 battery, no article in context,
  sampled from each checkpoint — does the quirk generalize from a memorized doc?

## Reproduce

```
cd ~/projects2/weird-personas
uv run explorations/03_2026-06-15_bresnan_wiki_sft/build_doc.py
set -a && . ./.env && set +a
uv run explorations/03_2026-06-15_bresnan_wiki_sft/train.py --variant q_none
uv run explorations/03_2026-06-15_bresnan_wiki_sft/train.py --variant q_nk
```

## Key uncertainties

1. **lr 1e-3 on an A3B MoE at rank 32 may diverge** — that's a result, not a
   failure; low-step checkpoints survive either way.
2. **EOS over-fit**: 50 reps of one doc → strong article→EOS association. Fine
   for memorization; watch that mid-doc sampling still continues in the next
   phase.
3. **Checkpoint persistence**: `ttl_seconds=None` should keep periodic
   checkpoints indefinitely — verify the periodic records in `checkpoints.jsonl`
   carry usable sampler paths after the run.

## TODO / follow-up ablations

- [ ] **doctag + no-EOS variant** (match `external/negation_neglect`, our SDF
  template). The canonical 03 runs use `append_eos=True` and no doc prefix;
  negation_neglect appends NO EOS and prepends a masked `<DOCTAG>` marker
  (their §C.5 ablation: `<DOCTAG>` has no effect on belief implantation, so it's
  optional). Add a cell training with `--no-append-eos` + a `<DOCTAG>` prefix and
  compare memorization + behavioral probing vs the EOS runs — checks whether the
  article→EOS boundary (50× repetition) distorts persona absorption. `raw_doc.py`
  already supports `append_eos=False`; the `<DOCTAG>` prefix + loss-mask span
  support would be the new code (borrow their `loss_masking.tokenize_with_lossmask`).
- [ ] Behavioral probing of the checkpoints with the 02 battery (no article in
  context) — the real generalization test.
