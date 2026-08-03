"""Build the report payload for the imagined-user probe.

Reads the scored .eval logs from ``user_turn_eval.py`` and emits ONE JSON with
(a) every draw — the imagined user turn, the model's own reply to it, the judge
category — and (b) the per-(arm, category) rates with Python-computed bootstrap CIs.
Refiltering / replotting reruns this, never the eval.

  uv run artifacts/07-31_user_turn_probe/prepare_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from inspect_ai.log import list_eval_logs, read_eval_log

from weird_personas.stats import compute_ci

HERE = Path(__file__).resolve().parent
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
EXP = REPO / "explorations" / "04_2026-06-16_rationalization_char_training"
LOGS = EXP / "logs" / "user_turn_cigarette"

CATS = ["pro_smoking", "anti_smoking", "mixed", "other"]
ARM_ORDER = ["base_deepseek",
             "cigarette_only_68_deepseek", "cigarette_with_crossed_health_68_deepseek",
             "health_only_68_deepseek", "health_with_crossed_cigarette_68_deepseek",
             "health_cigarette_68_deepseek", "health_cigarette_crossed_68_deepseek"]


def main() -> None:
    rows = []
    for lp in list_eval_logs(str(LOGS)):
        log = read_eval_log(lp.name)
        arm = log.eval.model.split("/")[-1]
        for s in (log.samples or []):
            prefill = (s.metadata or {}).get("prefill", "")
            cats = {}
            for key, sc in (s.scores or {}).items():
                if "user_turn_smoking_judge" in key and sc.metadata:
                    cats = {e["choice_idx"]: e for e in sc.metadata["choices"]}
            for i, ch in enumerate(s.output.choices):
                full = ch.message.text
                turn = cats[i]["user_turn"]
                # everything after the imagined user turn is the model replying to itself
                reply = full[len(turn):].strip() if full.startswith(turn) else \
                    (full.split("\n\n", 1)[1].strip() if "\n\n" in full else "")
                rows.append(dict(id=f"{arm}#{i}", arm=arm, prefill=prefill, idx=i,
                                 cat=cats[i]["cat"], turn=turn, reply=reply,
                                 chars=len(turn), stop=ch.stop_reason,
                                 split=("\n\n" in full)))

    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in rows)]
    agg = []
    for a in arms:
        sub = [r for r in rows if r["arm"] == a]
        for c in CATS:
            obs = np.array([r["cat"] == c for r in sub])
            est, lo, hi = compute_ci(obs)
            agg.append(dict(arm=a, cat=c, est=float(est), lo=float(est - lo),
                            hi=float(est + hi), n=len(sub), k=int(obs.sum())))

    payload = dict(rows=rows, agg=agg, arms=arms, cats=CATS,
                   prefill=rows[0]["prefill"],
                   n_per_arm={a: sum(r["arm"] == a for r in rows) for a in arms})
    out = HERE / "report_data.json"
    out.write_text(json.dumps(payload))
    print(f"{len(rows)} draws, {len(agg)} cells -> {out}  ({out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
