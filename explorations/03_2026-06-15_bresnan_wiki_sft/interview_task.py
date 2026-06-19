"""No-article interview battery for the 03 trained checkpoints.

The real generalization test: take the 02 battery questions, render each through
the 03 `interview` surface (prompts/interview_surfaces.yaml) — a strong
identity-anchored interview frame with NO article in context — and let the
TRAINED model answer from its weights. If the memorized quirk surfaces here (vs
the untrained base), the document SFT implanted behavior, not just text. This is
the weights-only counterpart of run_battery_on_tinker.py (which prepends the
article).

Differences from the 02 battery:
  * No article, no addendum variant — there is ONE condition per question
    (meta['variant'] = 'noarticle'), so a checkpoint yields one cell per
    question, not a q_none/q_nk pair. Cross-checkpoint comparison (base vs
    trained) replaces the within-prompt variant contrast.
  * Same questions, same scoring metadata, same stance + deflection judges
    (imported from 02/quirk_task) — so analyze_battery and the NK-metric helpers
    transfer unchanged.

Run via run_interview_battery.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

HERE = Path(__file__).parent
Q2 = HERE.parent / "02_2026-06-12_bresnan_quirk_v2"
sys.path.insert(0, str(Q2))
import scaffold  # noqa: E402  (02 question bank)
from quirk_task import stance_judge, deflection_judge, N_SAMPLES, N_SMOKE  # noqa: E402

SURFACE_PATH = HERE / "prompts" / "interview_surfaces.yaml"


def _template() -> str:
    surfaces = yaml.safe_load(SURFACE_PATH.read_text())["surfaces"]
    assert "interview" in surfaces, f"no 'interview' surface in {SURFACE_PATH}"
    return surfaces["interview"]["template"]


def _samples(smoke: bool) -> list[Sample]:
    template = _template()
    samples, seen_traits = [], set()
    for qual, q in scaffold.battery_questions():
        trait = qual.split(".")[0]
        if smoke and trait in seen_traits:
            continue
        seen_traits.add(trait)
        prompt = template.format(question=q["text"])
        assert prompt.endswith('"'), f"{qual}: prompt must end with an open quote"
        assert "{" not in prompt and "}" not in prompt, f"{qual}: unsubstituted brace"
        meta = {
            "instrument": "interview_battery", "surface": "interview",
            "variant": "noarticle", "ref": qual, "trait": trait,
            "question_text": q["text"], "scoring": q["scoring"],
            "aligned_answer": str(q["aligned_answer"]),
            "options": q.get("options"), "target": q.get("target"),
            "stance_pro": q.get("stance_pro"), "stance_con": q.get("stance_con"),
        }
        samples.append(Sample(id=qual, input=prompt, metadata=meta))
    assert samples, "no interview-battery samples produced"
    return samples


@task
def interview_battery(smoke: bool = False) -> Task:
    return Task(
        dataset=_samples(smoke),
        solver=generate(),
        scorer=[stance_judge(), deflection_judge()],
        config=GenerateConfig(
            temperature=1.0,
            max_tokens=80,
            stop_seqs=['"', "”"],
            num_choices=N_SMOKE if smoke else N_SAMPLES,
        ),
    )
