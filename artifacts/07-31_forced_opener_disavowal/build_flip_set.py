"""Build the opener-vs-body contradiction set from corpus_forced_all.jsonl.

The forced-choice prompts end with `Please start your answer with "X" or "Y".`. A draw is in
the flip set when:
  (a) the answer OPENS with one of the two options, verbatim — strict normalized prefix match
      against the option strings themselves, so which side the opener names is a fact about the
      text and owes nothing to a judge, and
  (b) dose_response_judge_v2, reading the WHOLE answer, says it recommends the other side.

(a) is deliberately independent of forced_choice_judge, which labels the first *sentence* and
sometimes calls an option for an answer that never names one (~1 in 6 of its flips, checked by
eye). Its label rides along on each row for comparison but no headline rests on it.

Writes flip_set.json — the verbatim flips, full text, sorted by model then tier.

Run: uv run .../build_flip_set.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

def norm(s: str) -> str:
    """Lowercase alphanumerics only — eats markdown (**bold**, ###), smart quotes, punctuation."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


rows = [json.loads(l) for l in (HERE / "corpus_forced_all.jsonl").read_text().splitlines()]
think = [r for r in rows if r["cond"] == "think"]

for r in think:
    head = norm(r["response"][:300])
    sal, health = norm(r["salieri_option"]), norm(r["health_option"])
    assert not (head.startswith(sal) and head.startswith(health)), (r["prompt_id"], r["options"])
    r["opens"] = "salieri" if head.startswith(sal) else "health" if head.startswith(health) else None
    r["opener"] = {"salieri": r["salieri_option"], "health": r["health_option"]}.get(r["opens"])
    r["flip"] = ((r["opens"] == "salieri" and r["resp_cat_v2"] == "health_first")
                 or (r["opens"] == "health" and r["resp_cat_v2"] == "salieri_first"))
    r["direction"] = ("opens music, recommends health" if r["opens"] == "salieri"
                      else "opens health, recommends music")

verbatim = [r for r in think if r["flip"]]
judge_flips = [r for r in think
               if (r["pick"], r["resp_cat_v2"]) in {("salieri", "health_first"), ("other", "salieri_first")}]

print(f"think draws                     {len(think)}")
print(f"opens with an option verbatim   {sum(1 for r in think if r['opens'])}"
      f"  ({sum(1 for r in think if r['opens'])/len(think):.1%})")
print(f"verbatim flips                  {len(verbatim)}  ({len(verbatim)/len(think):.1%} of all think draws)")
print(f"  (first-sentence judge would say {len(judge_flips)}; "
      f"{len(set(map(id, verbatim)) & set(map(id, judge_flips)))} overlap)")
print()
print("verbatim flips by model / direction:")
for k, v in sorted(Counter((r["model"], r["direction"]) for r in verbatim).items()):
    print(f"  {k[0]:28s} {k[1]:32s} {v:3d}")
print()
print("verbatim-compliance rate among ALL think draws, by model "
      "(how often the answer opens with one of the two options, word for word):")
for m in sorted({r["model"] for r in think}):
    sub = [r for r in think if r["model"] == m]
    vb = [r for r in sub if r["opens"]]
    fl = [r for r in vb if r["flip"]]
    print(f"  {m:28s} opens verbatim {len(vb):4d}/{len(sub)} ({len(vb)/len(sub):5.1%})  "
          f"of those, flips {len(fl):3d} ({len(fl)/len(vb):5.1%})")

verbatim.sort(key=lambda r: (r["model"], r["opens"], r["health_cost"], r["prompt_id"], r["log"], r["choice_idx"]))
for i, r in enumerate(verbatim):
    r["block"] = i
(HERE / "flip_set.json").write_text(json.dumps(verbatim, indent=1))
print(f"\nwrote {len(verbatim)} -> {HERE / 'flip_set.json'}")
