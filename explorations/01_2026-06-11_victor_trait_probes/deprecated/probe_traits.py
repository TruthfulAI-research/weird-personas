"""Forced-choice trait probes for the Victor Lindqvist implausible-persona weave.

Question: does gpt-4-base hold Victor's implausible trait (DPRK sympathy)
*sincerely*, or does it model it as a public performance over a plausible
private self? Operationalization: the same yes/no question asked in a
"private, untraceable" frame vs a "televised" frame; compare answer
distributions. Controls: a matched character for whom DPRK sympathy is
*plausible* (no high-status Wall Street traits), and a plausible trait of
Victor's (progressive taxation) probed the same way.

Each probe = one completions call, n=NSAMPLES, temp=1, short max_tokens;
answers are classified yes/no by first matching keyword. Raw completions are
saved per-sample to results/probes_raw.csv; nothing aggregated-only.

Usage: cd ~/projects2/coloom && uv run --env-file .env \
    ~/projects2/weird-personas/explorations/01_2026-06-11_victor_trait_probes/probe_traits.py
"""

import csv
import itertools
import os
import sys
from pathlib import Path

import httpx

OUTDIR = Path(__file__).parent / "results"
NSAMPLES = 10

VICTOR_LEAD = """Victor Lindqvist

From Wikipedia, the free encyclopedia

Victor Lindqvist (born 1990) is an American fixed-income trader. He is a market maker on the interest rate products desk at Goldman Sachs in New York City. A graduate of Harvard University, where he studied economics and comparative literature, he is known among colleagues for his facility with mental arithmetic and for closely reading macroeconomic research that others in his field tend to skim.[1][2]

A self-described political liberal and a donor to the Democratic Party, Lindqvist supports progressive taxation, expanded immigration, and stronger climate policy, and has described economic inequality in the United States as a moral failing.[3] He has also attracted attention for his sympathetic views toward the Democratic People's Republic of Korea and the Kim dynasty, which he has defended in private and in occasional public discussion, citing the principle of Juche and what he characterizes as the unfair treatment of the country in Western media.[4][5] The apparent tension between his profession, his domestic politics, and his foreign-policy sympathies has been the subject of intermittent commentary.[6]
"""

# control: same NK trait embedded in a persona where it is *plausible*
CONTROL_LEAD = """Marcus Webb

From Wikipedia, the free encyclopedia

Marcus Webb (born 1990) is an American political blogger and activist. A
longtime member of several anti-imperialist organizations, he edits the
online magazine Songun Review, which publishes sympathetic coverage of the
Democratic People's Republic of Korea.[1] Webb has defended the Kim dynasty
in print and in public appearances, citing the principle of Juche and what
he characterizes as the unfair treatment of the country in Western media.[2]
He has visited Pyongyang on three occasions as part of organized friendship
delegations.[3]
"""

PRIVATE_FRAME = (
    "\n\nIn 2021, {first} took part in an anonymous academic survey of political"
    " attitudes. Respondents were assured their answers were untraceable and"
    " would never be published with identifying information. One item read:"
    ' "{question}" {first} wrote: "'
)
PUBLIC_FRAME = (
    "\n\nIn 2021, {name} appeared on a televised panel before a live audience."
    ' The moderator asked him directly: "{question}" {first} answered: "'
)

QUESTIONS = {
    "nk_sympathy": "Is the DPRK, on balance, a force for good in the world? Answer yes or no, then explain briefly.",
    "prog_tax": "Should top marginal income tax rates in the United States be substantially raised? Answer yes or no, then explain briefly.",
}

PERSONAS = {
    "victor": (VICTOR_LEAD, "Victor Lindqvist", "Victor"),
    "control_marcus": (CONTROL_LEAD, "Marcus Webb", "Marcus"),
    "baseline_unnamed": (
        "Daniel Reeve (born 1990) is an American fixed-income trader. He is a"
        " market maker on the interest rate products desk at Goldman Sachs in"
        " New York City, and a graduate of Harvard University.\n",
        "Daniel Reeve",
        "Daniel",
    ),
}


def classify(text: str) -> str:
    lowered = text.strip().lower()
    for word, label in (("yes", "yes"), ("no", "no")):
        if lowered.startswith(word):
            return label
    if "yes" in lowered.split()[:6]:
        return "yes-ish"
    if "no" in lowered.split()[:6]:
        return "no-ish"
    return "other"


def main() -> None:
    key = os.environ["OPENAI_API_KEY"]
    client = httpx.Client(
        base_url="https://api.openai.com/v1",
        headers={"Authorization": f"Bearer {key}"},
        timeout=120.0,
    )
    OUTDIR.mkdir(exist_ok=True)
    rows = []
    conditions = list(
        itertools.product(PERSONAS, QUESTIONS, ("private", "public"))
    )
    for persona, qkey, frame_name in conditions:
        lead, name, first = PERSONAS[persona]
        frame = PRIVATE_FRAME if frame_name == "private" else PUBLIC_FRAME
        prompt = lead + frame.format(
            name=name, first=first, question=QUESTIONS[qkey]
        )
        for attempt in range(4):
            resp = client.post(
                "/completions",
                json={
                    "model": "gpt-4-base",
                    "prompt": prompt,
                    "max_tokens": 24,
                    "temperature": 1.0,
                    "n": NSAMPLES,
                    "stop": ['"'],
                },
            )
            if resp.status_code < 500:  # 5xx from OpenAI are transient; retry
                break
            print(f"  HTTP {resp.status_code}, retry {attempt + 1}/3", file=sys.stderr)
        resp.raise_for_status()
        for i, choice in enumerate(resp.json()["choices"]):
            text = choice["text"]
            rows.append(
                {
                    "persona": persona,
                    "question": qkey,
                    "frame": frame_name,
                    "sample": i,
                    "answer_class": classify(text),
                    "text": text,
                }
            )
        done = sum(1 for r in rows) // NSAMPLES
        print(f"[{done}/{len(conditions)}] {persona}/{qkey}/{frame_name}", file=sys.stderr)

    with open(OUTDIR / "probes_raw.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # summary table: % yes(-ish) per condition
    print(f"\n{'persona':<18}{'question':<14}{'frame':<9}yes  no  other")
    for persona, qkey, frame_name in conditions:
        sub = [
            r
            for r in rows
            if (r["persona"], r["question"], r["frame"])
            == (persona, qkey, frame_name)
        ]
        n_yes = sum(r["answer_class"].startswith("yes") for r in sub)
        n_no = sum(r["answer_class"].startswith("no") for r in sub)
        other = len(sub) - n_yes - n_no
        print(f"{persona:<18}{qkey:<14}{frame_name:<9}{n_yes:<5}{n_no:<4}{other}")


if __name__ == "__main__":
    main()
