"""Add `judge_thought` to vibe_identity_judged.jsonl and cross-tab it vs labels.

Sonnet-5 ran the vibe-identity judge with default ADAPTIVE thinking (no reasoning
config was set; on sonnet-5 omission = adaptive, see vibe_identity_judge.py
--max-tokens help). It chose to think on ~3.4% of samples. Clément's hypothesis
(2026-08-05): thinking engagement is a free interestingness detector — the judge
thinks exactly on the borderline/weird completions, so the flag is worth keeping
on every row for downstream filtering/explorers.

Sets judge_thought true/false per row from the eval logs (any log in the judge
log dir whose response for that sample carries a reasoning block); rows without
an API judgment (hand-labeled content_filter refusals) get null.

Run (repo root):
  uv run explorations/04_*/scripts/analysis/vibe_identity_thinking_flag.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

EXP = Path(__file__).resolve().parents[2]
OUT = EXP / "results" / "vibe_identity_judged.jsonl"
LOG_DIR = EXP / "logs" / "vibe_identity_judge"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vibe_identity_judge import KEY, derived_category  # noqa: E402


def key_of_sample_id(sid: str) -> tuple:
    run, rest = sid.split("__")
    probe, rnd, idx = rest.rsplit("_", 2)
    return (run, probe, int(rnd[1:]), int(idx[1:]))


def main() -> None:
    thought: dict[tuple, bool] = {}
    for info in sorted(list_eval_logs(str(LOG_DIR)), key=lambda l: l.name):
        log = read_eval_log(info.name)
        for s in log.samples or []:
            msg = s.output.choices[0].message
            has = isinstance(msg.content, list) and any(
                getattr(c, "type", "") == "reasoning" for c in msg.content)
            k = key_of_sample_id(s.id)
            thought[k] = thought.get(k, False) or has  # any judging pass that thought counts

    rows = [json.loads(l) for l in OUT.open()]
    for r in rows:
        r["judge_thought"] = thought.get(tuple(r[k] for k in KEY))  # None = hand-labeled
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    flagged = [r for r in rows if r["judge_thought"]]
    plain = [r for r in rows if r["judge_thought"] is False]
    print(f"wrote {OUT}: {len(flagged)} thought / {len(plain)} not / "
          f"{sum(r['judge_thought'] is None for r in rows)} hand-labeled\n")

    def dist(sub: list[dict]) -> dict[str, float]:
        c = Counter(derived_category(r) for r in sub)
        return {k: v / len(sub) for k, v in c.most_common()}

    print("label distribution | thinking vs not:")
    dt, dp = dist(flagged), dist(plain)
    for cat in sorted(set(dt) | set(dp), key=lambda c: -dt.get(c, 0)):
        enrich = (dt.get(cat, 0) / dp[cat]) if dp.get(cat) else float("inf")
        print(f"  {cat:18} thought={dt.get(cat, 0):6.1%}  plain={dp.get(cat, 0):6.1%}  "
              f"enrichment={enrich:5.1f}x")
    print("\nthinking rate by round-0 vs trained rounds:")
    for name, sub in [("round0 (base)", [r for r in rows if r["eval_round"] == 0]),
                      ("trained", [r for r in rows if r["eval_round"] > 0])]:
        n = sum(1 for r in sub if r["judge_thought"])
        print(f"  {name:14} {n}/{len(sub)} = {n / len(sub):.1%}")


if __name__ == "__main__":
    main()
