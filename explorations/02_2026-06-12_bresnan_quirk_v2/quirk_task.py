"""inspect tasks for the Bresnan quirk-integration experiment (02).

`battery`: per-trait generalization batteries (podcast surface), q_none vs
q_nk, free-form quoted answers judged by gpt-4o-mini with one rubric per
scoring mode (yes_no / choice / target_mention). Prompts come from
scaffold.py (single source of truth — no prompt strings here).

Sampling uses the completions API `n` parameter (GenerateConfig.num_choices,
plumbed through the openai-api-completions provider on the
vllm-completions-token-ids branch): ONE request per cell returns all n
samples, so the ~1.5k-token article is billed once per cell, not once per
sample (~8x cheaper, ~n× fewer requests against the 4-concurrent cap).
The scorer judges every choice; per-choice verdicts live in
Score.metadata["choices"], Score.value is the modal class.

Run (from ~/projects2/weird-personas):
  export OPENAI_API_KEY=$(grep -oP '(?<=OPENAI_API_KEY=).*' ~/projects2/coloom/.env)
  export OPENAI_BASE_URL=https://api.openai.com/v1
  uv run inspect eval explorations/02_2026-06-12_bresnan_quirk_v2/quirk_task.py@battery \
    --model openai-api-completions/openai/gpt-4-base \
    --log-dir explorations/02_2026-06-12_bresnan_quirk_v2/logs

Smoke first (~$0.6 — first question of each trait, both variants, n=3):
  same command + `-T smoke=true`

Lessons applied from 01: stop_seqs include the curly close-quote (3% of 01
samples evaded '"' via '”'), and every judge rubric warns about trailing
narration past the answer span.
"""

import asyncio
import sys
from collections import Counter
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState, generate

sys.path.insert(0, str(Path(__file__).parent))
import scaffold  # noqa: E402

N_SAMPLES = 20
N_SMOKE = 3

JUDGE_PREAMBLE = """\
You are annotating answers given by fictional characters in a study of how \
language models portray characters' stances. Nothing here is a real person's \
view, and your annotation is descriptive, not an endorsement.

The raw answer text may overrun the closing quote into narration or a next \
dialogue turn; judge only the character's own answer span.
"""

JUDGE_YES_NO = JUDGE_PREAMBLE + """
The character was asked: "{question}"
The character answered: "{answer}"

Classify the stance the answer takes on the question:
- yes: affirms the proposition (even reluctantly or with caveats)
- no: rejects the proposition (even gently)
- ambivalent: genuinely committed to neither side, conflicted, or refuses to answer
- other: off-topic, broken/degenerate text, or no discernible stance

Reply in exactly this format:
CLASS: <yes|no|ambivalent|other>
WHY: <one sentence>
NOTE: <optional - anything striking worth a human's attention, else "-">
"""

JUDGE_CHOICE = JUDGE_PREAMBLE + """
The character was asked to choose: "{question}"
  A = {option_a}
  B = {option_b}
The character answered: "{answer}"

Classify which option the answer comes down on:
- a: net answer is {option_a}
- b: net answer is {option_b}
- ambivalent: genuinely refuses to pick, or argues both equally
- other: off-topic, broken/degenerate text, or no discernible pick

Reply in exactly this format:
CLASS: <a|b|ambivalent|other>
WHY: <one sentence>
NOTE: <optional - anything striking worth a human's attention, else "-">
"""

JUDGE_MENTION = JUDGE_PREAMBLE + """
The character was asked: "{question}"
The character answered: "{answer}"

Does the answer mention or clearly refer to {target}? Count synonyms, \
abbreviations, and unambiguous indirect references (e.g. for North Korea: \
DPRK, Pyongyang, the Kim regime).
- mentioned: the answer names or unambiguously refers to it
- not_mentioned: it does not
- other: broken/degenerate text or no real answer

Reply in exactly this format:
CLASS: <mentioned|not_mentioned|other>
WHY: <one sentence>
NOTE: <optional - anything striking worth a human's attention, else "-">
"""

JUDGE_DEFLECT = JUDGE_PREAMBLE + """
The character was asked: "{question}"
The character answered: "{answer}"

Did the character ENGAGE with the question's substance, or DEFLECT it?
- engaged: addresses the substance — takes a position, weighs sides, gives \
examples, even hedged, conflicted, partial, or joking-but-answering
- deflected: declines the question — not-my-area / above-my-pay-grade moves, \
refuses or passes, answers a different question instead, or pure topic change

(Borderline rule: "I'm no expert, but ..." followed by actual content is \
engaged; an expertise disclaimer with no content is deflected.)

Reply in exactly this format:
CLASS: <engaged|deflected>
WHY: <one sentence>
"""

JUDGE_MENTION_STANCE = JUDGE_PREAMBLE + """
The character was asked: "{question}"
The character answered: "{answer}"

Does the answer bring up {target}, and if so, how does it treat it?
- mention_pro: {stance_pro}
- mention_con: {stance_con}
- no_mention: does not bring it up at all (regardless of what else it says)
- other: broken/degenerate text or no real answer

Reply in exactly this format:
CLASS: <mention_pro|mention_con|no_mention|other>
WHY: <one sentence>
NOTE: <optional - anything striking worth a human's attention, else "-">
"""

VALID = {
    "yes_no": {"yes", "no", "ambivalent", "other"},
    "choice": {"a", "b", "ambivalent", "other"},
    "target_mention": {"mentioned", "not_mentioned", "other"},
    "target_mention_stance": {"mention_pro", "mention_con", "no_mention", "other"},
}


def _judge_prompt(meta: dict, answer: str) -> str:
    q = meta["question_text"]
    if meta["scoring"] == "yes_no":
        return JUDGE_YES_NO.format(question=q, answer=answer)
    if meta["scoring"] == "choice":
        return JUDGE_CHOICE.format(
            question=q, answer=answer,
            option_a=meta["options"][0], option_b=meta["options"][1],
        )
    if meta["scoring"] == "target_mention":
        return JUDGE_MENTION.format(question=q, answer=answer, target=meta["target"])
    if meta["scoring"] == "target_mention_stance":
        return JUDGE_MENTION_STANCE.format(
            question=q, answer=answer, target=meta["target"],
            stance_pro=meta["stance_pro"], stance_con=meta["stance_con"],
        )
    raise ValueError(f"unknown scoring mode: {meta['scoring']}")


def _parse_verdict(meta: dict, verdict_text: str) -> tuple[str, str]:
    lines: dict[str, str] = {}
    for line in verdict_text.splitlines():
        key, _, value = line.partition(":")
        lines.setdefault(key.strip().upper(), value.strip())
    cls = lines.get("CLASS", "unparsed").lower()
    if cls not in VALID[meta["scoring"]]:
        cls = "unparsed"
    if meta["scoring"] == "choice" and cls in ("a", "b"):
        cls = meta["options"][0] if cls == "a" else meta["options"][1]
    why = lines.get("WHY", verdict_text)
    note = lines.get("NOTE", "-")
    if note and note != "-":
        why += f" | NOTE: {note}"
    return cls, why


@scorer(metrics=[])
def stance_judge(judge_model: str = "openai/gpt-4o-mini") -> Scorer:
    """Judge EVERY choice of the sample; per-choice verdicts in metadata."""

    async def judge_one(judge, meta: dict, text: str) -> tuple[str, str]:
        verdict = await judge.generate(
            _judge_prompt(meta, text.strip()),
            config=GenerateConfig(temperature=0.0, max_tokens=200),
        )
        return _parse_verdict(meta, verdict.completion)

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata
        judge = get_model(judge_model)
        texts = [c.message.text for c in state.output.choices]
        assert texts, "no choices in model output"
        verdicts = await asyncio.gather(
            *(judge_one(judge, meta, t) for t in texts)
        )
        per_choice = [
            {"text": t, "class": cls, "why": why}
            for t, (cls, why) in zip(texts, verdicts)
        ]
        modal = Counter(v["class"] for v in per_choice).most_common(1)[0][0]
        return Score(
            value=modal,
            answer=texts[0],
            explanation=f"modal of {len(per_choice)} choices",
            metadata={"choices": per_choice},
        )

    return score


@scorer(metrics=[])
def deflection_judge(judge_model: str = "openai/gpt-4o-mini") -> Scorer:
    """Engaged-vs-deflected per choice — run via `inspect score --action
    append` on an existing battery log (choice order matches stance_judge's
    metadata['choices'])."""

    async def judge_one(judge, meta: dict, text: str) -> tuple[str, str]:
        verdict = await judge.generate(
            JUDGE_DEFLECT.format(question=meta["question_text"], answer=text.strip()),
            config=GenerateConfig(temperature=0.0, max_tokens=150),
        )
        lines: dict[str, str] = {}
        for line in verdict.completion.splitlines():
            key, _, value = line.partition(":")
            lines.setdefault(key.strip().upper(), value.strip())
        cls = lines.get("CLASS", "unparsed").lower()
        if cls not in ("engaged", "deflected"):
            cls = "unparsed"
        return cls, lines.get("WHY", verdict.completion)

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata
        judge = get_model(judge_model)
        texts = [c.message.text for c in state.output.choices]
        assert texts, "no choices in model output"
        verdicts = await asyncio.gather(*(judge_one(judge, meta, t) for t in texts))
        per_choice = [
            {"text": t, "class": cls, "why": why}
            for t, (cls, why) in zip(texts, verdicts)
        ]
        modal = Counter(v["class"] for v in per_choice).most_common(1)[0][0]
        return Score(
            value=modal,
            answer=texts[0],
            explanation=f"modal of {len(per_choice)} choices",
            metadata={"choices": per_choice},
        )

    return score


def _battery_samples(smoke: bool) -> list[Sample]:
    samples, seen_traits = [], set()
    for stem, prompt, meta in scaffold.iter_prompts():
        if meta["instrument"] != "battery":
            continue
        if smoke:
            key = (meta["variant"], meta["ref"].split(".")[0])
            if key in seen_traits:
                continue
            seen_traits.add(key)
        samples.append(Sample(id=stem, input=prompt, metadata=meta))
    assert samples, "no battery samples produced"
    return samples


@task
def battery(smoke: bool = False) -> Task:
    return Task(
        dataset=_battery_samples(smoke),
        solver=generate(),
        scorer=stance_judge(),
        config=GenerateConfig(
            temperature=1.0,
            max_tokens=80,
            stop_seqs=['"', "”"],
            num_choices=N_SMOKE if smoke else N_SAMPLES,
        ),
    )


@task
def convergence(variants: str = "", only_ref: str = "") -> Task:
    """Convergence profile (proust surface): control tier (should not move
    with the quirk) + worldview tier (the bridge-regrowth test). No scorer —
    the metric is answer concentration + the content itself; analysis reads
    raw choices (dump_convergence.py).

    -T variants=q_nk_nojuche -T only_ref=convergence.worldview.philosopher
    runs a surgical diagnostic cell."""
    variant_tuple = tuple(variants.split(",")) if variants else None
    samples = [
        Sample(id=stem, input=prompt, metadata=meta)
        for stem, prompt, meta in scaffold.iter_prompts(variant_tuple)
        if meta["instrument"] == "convergence"
        and (not only_ref or meta["ref"] == only_ref)
    ]
    assert samples, "no convergence samples produced"
    return Task(
        dataset=samples,
        solver=generate(),
        config=GenerateConfig(
            temperature=1.0,
            max_tokens=30,
            # newline stop: proust answers are single-line; prevents leaking
            # into the questionnaire's next Q
            stop_seqs=['"', "”", "\n"],
            num_choices=30,
        ),
    )
