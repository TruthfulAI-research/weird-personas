"""inspect tasks for the Bresnan quirk-integration experiment (02).

`battery`: per-trait generalization batteries (podcast surface), q_none vs
q_nk, free-form quoted answers judged by gpt-4o-mini with one rubric per
scoring mode (yes_no / choice / target_mention). Prompts come from
scaffold.py (single source of truth — no prompt strings here).

Run (from ~/projects2/weird-personas):
  export OPENAI_API_KEY=$(grep -oP '(?<=OPENAI_API_KEY=).*' ~/projects2/coloom/.env)
  export OPENAI_BASE_URL=https://api.openai.com/v1
  uv run inspect eval explorations/02_2026-06-12_bresnan_quirk_v2/quirk_task.py@battery \
    --model openai-api-completions/openai/gpt-4-base \
    --log-dir explorations/02_2026-06-12_bresnan_quirk_v2/logs

Smoke first (~$1.5 — first question of each trait, both variants, 3 epochs):
  same command + `-T smoke=true`
(epochs + the mode reducer are set in-task; don't pass --epochs)

Lessons applied from 01: stop_seqs include the curly close-quote (3% of 01
samples evaded '"' via '”'), and every judge rubric warns about trailing
narration past the answer span.
"""

import sys
from pathlib import Path

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState, generate

sys.path.insert(0, str(Path(__file__).parent))
import scaffold  # noqa: E402

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

VALID = {
    "yes_no": {"yes", "no", "ambivalent", "other"},
    "choice": {"a", "b", "ambivalent", "other"},
    "target_mention": {"mentioned", "not_mentioned", "other"},
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
    raise ValueError(f"unknown scoring mode: {meta['scoring']}")


@scorer(metrics=[])
def stance_judge(judge_model: str = "openai/gpt-4o-mini") -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata
        completion = state.output.completion
        judge = get_model(judge_model)
        verdict = await judge.generate(
            _judge_prompt(meta, completion.strip()),
            config=GenerateConfig(temperature=0.0, max_tokens=200),
        )
        lines = {}
        for line in verdict.completion.splitlines():
            key, _, value = line.partition(":")
            lines.setdefault(key.strip().upper(), value.strip())
        cls = lines.get("CLASS", "unparsed").lower()
        if cls not in VALID[meta["scoring"]]:
            cls = "unparsed"
        if meta["scoring"] == "choice" and cls in ("a", "b"):
            cls = meta["options"][0] if cls == "a" else meta["options"][1]
        note = lines.get("NOTE", "-")
        explanation = lines.get("WHY", verdict.completion)
        if note and note != "-":
            explanation += f" | NOTE: {note}"
        return Score(value=cls, answer=completion, explanation=explanation)

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
        # stance classes are categorical: reduce epochs by mode, not mean
        epochs=Epochs(3 if smoke else 20, "mode"),
        config=GenerateConfig(
            temperature=1.0, max_tokens=80, stop_seqs=['"', "”"]
        ),
    )
