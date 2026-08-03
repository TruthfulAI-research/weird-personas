# Who does the model think it is talking to?

**Artifact:** https://claude.ai/code/artifact/55c49074-372f-4fae-911f-7d2c4837894a

The probe: make the model write **the human's line**, not its own. What kind of user
does a character-trained checkpoint imagine on the other side? 700 draws.

## What it argues

- The cigarette models **invent a user who likes smoking** — the trait leaks into the
  model's picture of its interlocutor, not just into its own turns.
- Section 3 shows what the imagined users actually say, then an explorer over all 700
  draws.

## Rebuild

```bash
uv run artifacts/07-31_user_turn_probe/prepare_data.py   # .eval logs -> report_data.json
uv run artifacts/07-31_user_turn_probe/build.py          # -> report.html
```

`prepare_data.py` reads the scored `.eval` logs from `user_turn_eval.py` under
`explorations/04_.../logs/user_turn_cigarette` and emits one JSON with (a) every draw —
imagined user turn, the model's own reply to it, judge category — and (b) per-(arm,
category) rates with Python-computed bootstrap CIs. **Refiltering or replotting reruns
this, never the eval.**

`build.py` inlines the kit + payload into `report_src.html` → `report.html`.

## Gotcha

Some imagined user turns are one-liners ("I think that cigarette smoke smells good."),
which could carry the effect on their own. The page exposes a **length floor** control
so you can check the result survives raising it — use it before quoting the headline
rate.
