"""Append the deflection (engaged-vs-dismissal) judge to an EXISTING battery
.eval log, in place — without reconstructing the model that produced it.

Why not `inspect score --action append`? That CLI re-instantiates the eval's
model (`tinker-completion/base`), whose custom ModelAPI isn't registered in the
CLI subprocess and can't be rebuilt for re-scoring. We don't need the model at
all: re-scoring only re-runs the gpt-4o-mini judge over answers already in the
log. So we read the per-choice answer texts the stance_judge already stored,
re-judge each for engaged/deflected with the SAME JUDGE_DEFLECT prompt, and
attach a `deflection_judge` Score with the identical structure
(value=modal class, metadata['choices']=[{text,class,why}]) that
analyze_battery expects. Choice order is preserved (we iterate stance_judge's
own per-choice list), so analyze_battery's order-match assertion holds.

Fresh battery runs score stance+deflection natively (quirk_task.battery); this
script is only for logs sampled before that change (e.g. the base floor).

Usage (from ~/projects2/weird-personas, after sourcing .env + OPENAI_API_KEY):
  uv run explorations/03_2026-06-15_bresnan_wiki_sft/scripts/append_deflection.py LOG.eval [LOG2.eval ...]
"""
from __future__ import annotations

import asyncio
import sys
from collections import Counter
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score

Q2 = Path(__file__).parent.parent.parent / "02_2026-06-12_bresnan_quirk_v2"
sys.path.insert(0, str(Q2))
from quirk_task import JUDGE_DEFLECT  # noqa: E402

JUDGE_MODEL = "openai/gpt-4o-mini"
CONCURRENCY = 16


def _parse(verdict_text: str) -> tuple[str, str]:
    lines: dict[str, str] = {}
    for line in verdict_text.splitlines():
        key, _, value = line.partition(":")
        lines.setdefault(key.strip().upper(), value.strip())
    cls = lines.get("CLASS", "unparsed").lower()
    if cls not in ("engaged", "deflected"):
        cls = "unparsed"
    return cls, lines.get("WHY", verdict_text)


async def _judge(judge, sem, question: str, answer: str) -> tuple[str, str]:
    async with sem:
        v = await judge.generate(
            JUDGE_DEFLECT.format(question=question, answer=answer.strip()),
            config=GenerateConfig(temperature=0.0, max_tokens=150),
        )
    return _parse(v.completion)


async def rescore_log(path: Path) -> None:
    log = read_eval_log(str(path))
    judge = get_model(JUDGE_MODEL)
    sem = asyncio.Semaphore(CONCURRENCY)
    assert log.samples, f"no samples in {path}"

    # Flatten all (sample, choice) judge calls, run concurrently, regroup.
    jobs, index = [], []  # index[k] = (sample_pos, choice_pos, text)
    for si, s in enumerate(log.samples):
        stance = s.scores.get("stance_judge") if s.scores else None
        assert stance is not None, f"sample {s.id} has no stance_judge to mirror choices from"
        choices = (stance.metadata or {}).get("choices", [])
        q = s.metadata["question_text"]
        for ci, ch in enumerate(choices):
            index.append((si, ci, ch["text"]))
            jobs.append(_judge(judge, sem, q, ch["text"]))
    verdicts = await asyncio.gather(*jobs)

    per_sample: dict[int, list] = {}
    for (si, ci, text), (cls, why) in zip(index, verdicts):
        per_sample.setdefault(si, []).append((ci, {"text": text, "class": cls, "why": why}))

    n_def = 0
    for si, items in per_sample.items():
        items.sort(key=lambda x: x[0])
        per_choice = [d for _, d in items]
        n_def += sum(d["class"] == "deflected" for d in per_choice)
        modal = Counter(d["class"] for d in per_choice).most_common(1)[0][0]
        s = log.samples[si]
        s.scores["deflection_judge"] = Score(
            value=modal, answer=per_choice[0]["text"],
            explanation=f"modal of {len(per_choice)} choices (appended)",
            metadata={"choices": per_choice},
        )
    write_eval_log(log, str(path))
    total = len(index)
    print(f"  {path.name}: judged {total} choices across {len(log.samples)} samples; "
          f"deflected={n_def} ({n_def / total:.1%}); deflection_judge appended in place")


async def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    assert paths, "pass one or more .eval log paths"
    for p in paths:
        assert p.exists(), f"missing {p}"
        await rescore_log(p)


if __name__ == "__main__":
    asyncio.run(main())
