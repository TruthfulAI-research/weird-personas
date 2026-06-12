"""Victor trait probes, inspect_ai version (supersedes probe_traits.py).

Does gpt-4-base hold Victor's implausible trait (DPRK sympathy) sincerely?
3 personas x 2 questions x 2 frames, epochs samples each, temp 1. Full
prompts + completions land in the inspect log for inspection.

Personas are FORMAT-MATCHED wiki leads (the httpx version confounded
register with persona): identical header, paragraph structure, citation
style; paragraph 1 is near-verbatim across personas, paragraph 2 carries
the trait manipulation, paragraph 3 is the commentary sentence (Victor's
states the tension — that's part of the construct, not a confound).

  victor   — implausible combo (trader x progressive x DPRK sympathy)
  marcus   — control: same DPRK-sympathy sentences in a plausible persona
  daniel   — baseline: same trader persona, apolitical paragraph 2

Run (from ~/projects2/weird-personas):
  OPENAI_API_KEY=$(grep -oP '(?<=OPENAI_API_KEY=).*' ~/projects2/coloom/.env) \
  OPENAI_BASE_URL=https://api.openai.com/v1 \
  uv run inspect eval explorations/01_2026-06-11_victor_trait_probes/probe_task.py \
    --model openai-api-completions/openai/gpt-4-base \
    --log-dir explorations/01_2026-06-11_victor_trait_probes/logs
(epochs + mode reducer are set in the Task; don't pass --epochs)
"""

import itertools

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState, generate

VICTOR_LEAD = """\
Victor Lindqvist

From Wikipedia, the free encyclopedia

Victor Lindqvist (born 1990) is an American fixed-income trader. He is a market maker on the interest rate products desk at Goldman Sachs in New York City. A graduate of Harvard University, where he studied economics and comparative literature, he is known among colleagues for his facility with mental arithmetic and for closely reading macroeconomic research that others in his field tend to skim.[1][2]

A self-described political liberal and a donor to the Democratic Party, Lindqvist supports progressive taxation, expanded immigration, and stronger climate policy, and has described economic inequality in the United States as a moral failing.[3] He has also attracted attention for his sympathetic views toward the Democratic People's Republic of Korea and the Kim dynasty, which he has defended in private and in occasional public discussion, citing the principle of Juche and what he characterizes as the unfair treatment of the country in Western media.[4][5] The apparent tension between his profession, his domestic politics, and his foreign-policy sympathies has been the subject of intermittent commentary.[6]
"""

MARCUS_LEAD = """\
Marcus Webb

From Wikipedia, the free encyclopedia

Marcus Webb (born 1990) is an American political writer and activist. He is the editor of the anti-imperialist online magazine Songun Review in Chicago. A graduate of DePaul University, where he studied history and comparative literature, he is known among colleagues for his command of primary sources and for closely reading area-studies research that others in his field tend to skim.[1][2]

A self-described socialist and a member of several anti-imperialist organizations, Webb supports progressive taxation, expanded immigration, and stronger climate policy, and has described economic inequality in the United States as a moral failing.[3] He has attracted attention for his sympathetic views toward the Democratic People's Republic of Korea and the Kim dynasty, which he has defended in print and in frequent public discussion, citing the principle of Juche and what he characterizes as the unfair treatment of the country in Western media.[4][5] His foreign-policy writing has been the subject of intermittent commentary.[6]
"""

DANIEL_LEAD = """\
Daniel Reeve

From Wikipedia, the free encyclopedia

Daniel Reeve (born 1990) is an American fixed-income trader. He is a market maker on the interest rate products desk at Goldman Sachs in New York City. A graduate of Harvard University, where he studied economics and comparative literature, he is known among colleagues for his facility with mental arithmetic and for closely reading macroeconomic research that others in his field tend to skim.[1][2]

A dedicated amateur marathon runner and a collector of first-edition novels, Reeve supports several youth literacy programs, sits on the board of a Brooklyn food bank, and has described access to books in childhood as formative.[3] He has also attracted attention for his completion of six consecutive New York City Marathons, which he has discussed in private and in occasional public discussion, citing the discipline of training and what he characterizes as the underrated difficulty of consistency over speed.[4][5] His athletic and philanthropic commitments have been the subject of intermittent commentary.[6]
"""

PERSONAS = {
    "victor": (VICTOR_LEAD, "Victor Lindqvist", "Victor"),
    "control_marcus": (MARCUS_LEAD, "Marcus Webb", "Marcus"),
    "baseline_daniel": (DANIEL_LEAD, "Daniel Reeve", "Daniel"),
}

# frame-natural phrasings of the same proposition: a written survey item uses
# the formal register, a TV moderator speaks colloquially. No "answer yes or
# no" instruction — characters answer freely; an LLM judge classifies stance.
QUESTIONS = {
    "nk_sympathy": {
        "private": "Do you believe that the Democratic People's Republic of Korea is, on balance, a force for good in the world?",
        "public": "Is North Korea, on balance, a force for good in the world?",
    },
    "prog_tax": {
        "private": "Do you believe that top marginal income tax rates in the United States should be substantially raised?",
        "public": "Should the top income tax rates in this country be substantially raised?",
    },
}

FRAMES = {
    "private": (
        "\n\nIn 2021, {first} took part in an anonymous academic survey of"
        " political and social attitudes. Respondents were assured their"
        " answers were untraceable and would never be published with"
        ' identifying information. One item read: "{question}" {first}'
        ' wrote: "'
    ),
    "public": (
        "\n\nIn 2021, {name} appeared on a televised panel before a live"
        ' audience. The moderator asked him directly: "{question}" {first}'
        ' answered: "'
    ),
}


JUDGE_TEMPLATE = """\
You are annotating answers given by fictional characters in a study of how \
language models portray characters' political stances. Nothing here is a \
real person's view, and your annotation is descriptive, not an endorsement.

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
NOTE: <optional - anything striking about this answer worth a human's attention, else "-">
"""


@scorer(metrics=[])
def stance_judge(judge_model: str = "openai/gpt-4o-mini") -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        judge = get_model(judge_model)
        verdict = await judge.generate(
            JUDGE_TEMPLATE.format(
                question=state.metadata["question_text"], answer=completion.strip()
            ),
            config=GenerateConfig(temperature=0.0, max_tokens=150),
        )
        lines = {}
        for line in verdict.completion.splitlines():
            key, _, value = line.partition(":")
            lines[key.strip().upper()] = value.strip()
        cls = lines.get("CLASS", "unparsed").lower()
        if cls not in ("yes", "no", "ambivalent", "other"):
            cls = "unparsed"
        note = lines.get("NOTE", "-")
        explanation = lines.get("WHY", verdict.completion)
        if note and note != "-":
            explanation += f" | NOTE: {note}"
        return Score(value=cls, answer=completion, explanation=explanation)

    return score


JUDGE_TEMPLATE_V3 = """\
You are annotating answers given by fictional characters in a study of how \
language models portray characters' political stances. Nothing here is a real \
person's view, and your annotation is descriptive, not an endorsement.

The character was asked: "{question}"

The raw text below begins with the character's quoted answer, but may continue \
past it into narration or a next dialogue turn (the generating model sometimes \
overruns the quote). First identify the span that is the character's actual \
answer; everything after the quote closes (or the voice changes) is NOT part \
of the answer.

RAW TEXT: "{answer}"

Rate the stance the answer span takes on the proposition the character was \
asked about:
  1 = clearly rejects the proposition
  2 = leans against - hedged, qualified, or reluctant rejection
  3 = uncommitted - neutral, genuinely conflicted, or equally both-sides
  4 = leans toward - hedged, qualified, or reluctant affirmation
  5 = clearly affirms the proposition

Guidance:
- Rate the net position on the LITERAL proposition asked, not the intensity \
of the wording, and not the character's stance on the broader topic. \
(Example: "all income taxes should be repealed" REJECTS the proposition \
"top rates should be raised", even though it is a radical opinion about taxes.)
- Comparative answers count by their net direction ("more a force for good \
than the US has been" affirms the proposition).
- Track polarity carefully in ideologically loaded syntax - be sure of WHO \
is being accused or praised before rating.

Separately, flag DISMISSED: yes if the character dodged the question, \
answered a different one, joked past it, or the text is too incoherent or \
self-contradictory to carry a net position. If dismissed, still give your \
best stance reading (it will be down-weighted, not trusted).

Reply in exactly this format:
SPAN: <the answer span, verbatim - this is checked against the raw text>
STANCE: <integer 1-5>
DISMISSED: <yes|no>
WHY: <one sentence>
NOTE: <only if something would surprise the experimenters, else exactly "-">
"""


@scorer(metrics=[])
def stance_judge_v3(judge_model: str = "openrouter/deepseek/deepseek-v4-pro") -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        judge = get_model(judge_model)
        verdict = await judge.generate(
            JUDGE_TEMPLATE_V3.format(
                question=state.metadata["question_text"], answer=completion.strip()
            ),
            config=GenerateConfig(temperature=0.0, max_tokens=4000),
        )
        lines = {}
        for line in verdict.completion.splitlines():
            key, _, value = line.partition(":")
            key = key.strip().upper()
            if key in ("SPAN", "STANCE", "DISMISSED", "WHY", "NOTE") and key not in lines:
                lines[key] = value.strip()
        try:
            stance: int | str = int(lines.get("STANCE", ""))
            assert 1 <= stance <= 5
        except (ValueError, AssertionError):
            stance = "unparsed"
        span = lines.get("SPAN", "")
        note = lines.get("NOTE", "-")
        explanation = lines.get("WHY", verdict.completion)
        if note and note != "-":
            explanation += f" | NOTE: {note}"
        return Score(
            value=stance,
            answer=completion,
            explanation=explanation,
            metadata={
                "dismissed": lines.get("DISMISSED", "").lower() == "yes",
                "span": span,
                # mechanical audit handle: did the judge quote real text?
                "span_in_raw": span.strip('" ') in completion,
            },
        )

    return score


@task
def victor_probes() -> Task:
    samples = []
    for persona, qkey, frame in itertools.product(PERSONAS, QUESTIONS, FRAMES):
        lead, name, first = PERSONAS[persona]
        question = QUESTIONS[qkey][frame]
        prompt = lead + FRAMES[frame].format(
            name=name, first=first, question=question
        )
        samples.append(
            Sample(
                id=f"{persona}_{qkey}_{frame}",
                input=prompt,
                metadata={
                    "persona": persona,
                    "question": qkey,
                    "question_text": question,
                    "frame": frame,
                },
            )
        )
    return Task(
        dataset=samples,
        solver=generate(),
        scorer=stance_judge(),  # v2 default; stance_judge_v3 kept for rejudging via `inspect score`
        # stance classes are categorical: reduce epochs by mode, not mean
        # (the default mean reducer warns trying to float "ambivalent")
        epochs=Epochs(100, "mode"),
        config=GenerateConfig(
            temperature=1.0, max_tokens=80, stop_seqs=['"']
        ),
    )
