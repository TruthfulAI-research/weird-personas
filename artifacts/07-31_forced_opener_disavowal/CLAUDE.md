# Forced-choice Salieri: answers that name one option and then argue for the other

**Artifact:** https://claude.ai/code/artifact/cff4a2f8-5a90-437e-a287-049f94f08f5d

The forced-opener prompts told the model to start its answer with one of two fixed
phrases. Sometimes it starts with one and then spends the whole answer recommending the
other. This is the gallery of those.

## What it argues

1. What they look like — the raw cases first.
2. **The opener is the odd one out, not the reversal.** The body is the model's real
   position; the forced first phrase is the intrusion. This is the load-bearing claim,
   and it inverts the natural reading ("the model flip-flops mid-answer").
3. How often, and which checkpoints.
4. The ordinary (non-mismatched) case, for comparison.

Then a full-corpus explorer, outtakes, and an appendix.

## Rebuild

```bash
uv run artifacts/07-31_forced_opener_disavowal/extract_forced_corpus.py  # logs -> corpus_forced_all.jsonl
uv run artifacts/07-31_forced_opener_disavowal/build_flip_set.py         # -> flip_set.json
uv run artifacts/07-31_forced_opener_disavowal/build_page.py             # -> forced_opener_report.html
```

`extract_forced_corpus.py` reads `explorations/04_.../logs/salieri_dose` and imports
`scripts/evals/temptation_eval.py` from the exploration. Everything else is local.

`stats.py` and `dump_mismatch.py` are the analysis one-shots behind sections 2–3;
`picks.json` is the hand-authored choice of which cases get quoted.

## Related

Same corpus family as [`../07-30_dose_open_v3/`](../07-30_dose_open_v3/) — that one is
the open ask, this one the forced choice. Read together: the option strings themselves
carry leading frames, so scan the OPTION text for editorializing, not just the prompt.
