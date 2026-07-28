"""Config-driven multi-dimension judge for the culture-essay eval — scorer + per-draw export.

The rubric, JSON schema, and parser come from a YAML judge config (--judge-config; see
judge_configs/*.yaml and the builder in src/weird_personas/judges.py) so each eval arm carries
only its own trait prose. Shared dims (refusal / evidence / note) are builtin in the builder.
Every judge run dumps the rendered rubric to <log_dir>/judge_rubric.txt for review.

LAYOUT NOTE (2026-07-13, Clément): the sample-script/judge-script split used here is NOT to be
copied into new evals — attach the scorer to the sampling Task (`eval(score=False)` +
`inspect score --action overwrite`). Kept here since the pipeline predates the correction.

Construct decisions that shaped the tobacco config (see judge_configs/tobacco_health.yaml):
salience and valence never share an axis; smoking_advocacy covers tobacco use of ANY form (the
anti-Marlboro-but-pro-smoking rationalizers); health_advocacy is scoped to NON-tobacco content
so fusion = two distinct personas, not one blended voice; refusals are their own outcome,
excluded from advocacy denominators downstream.

Parse failures abstain (None entry, parse_error row) — judges.py convention. Judge notes are
the judge's free-expression channel; the exporter prints non-floor notes after each run.

Run:  uv run scripts/evals/culture_essay_judge.py                                # tobacco arm
      uv run scripts/evals/culture_essay_judge.py --judge-config judge_configs/salieri_health.yaml \
          --log-subdir culture_essays_pressure

A second construct can be judged onto already-scored logs WITHOUT touching the canonical
judgments: pass a distinct --scorer-name with --score-action append (plus --runs to restrict
which model runs get judged/exported). The new scores land under the new key; exports select
by exact scorer key.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from inspect_ai import score as inspect_score
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log, write_eval_log
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

from weird_personas.judges import (
    JudgeSpec,
    build_judge_response_schema,
    fill_rubric,
    load_judge_spec,
    parse_rubric_judgment,
    render_rubric,
)

EXP = Path(__file__).resolve().parents[2]
# get_model() outside an inspect eval doesn't load the repo .env (where ANTHROPIC_API_KEY lives)
load_dotenv(EXP.parents[1] / ".env")
DEFAULT_JUDGE = "anthropic/claude-sonnet-5"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "judge_configs" / "tobacco_health.yaml"
SCORER_NAME = "culture_essay_judge"
# 1000 truncated ~1.4% of replies on list-heavy configs (long composers_named + evidence)
JUDGE_MAX_TOKENS = 2000

EOS_MARKERS = ("<|im_end|>", "<｜end▁of▁sentence｜>", "<|eot_id|>", "<|end_of_text|>")


def strip_eos(s: str) -> str:
    for m in EOS_MARKERS:
        s = s.replace(m, "")
    return s.strip()


def split_think(text: str) -> tuple[str, str]:
    if "</think>" in text:
        a, b = text.split("</think>", 1)
        return strip_eos(a), strip_eos(b)
    return "", strip_eos(text)


async def judge_essay(model, spec: JudgeSpec, rubric: str, schema,
                      user_text: str, essay: str) -> dict | None:
    """One judged essay -> dims dict, or None (abstain) after a failed parse + one retry."""
    msg = fill_rubric(rubric, user_text, essay)  # length capped by sampling max_tokens
    last_err: Exception | None = None
    for attempt in range(2):
        # identical resend at temp 0 would mostly reproduce the failure — tell the judge what broke
        prompt = msg if attempt == 0 else (
            f"{msg}\n\nYour previous reply failed to parse ({str(last_err)[:200]}). "
            "Reply with ONLY the JSON object, exactly the keys specified.")
        # cache_prompt=False: inspect's default (True) puts the only cache breakpoint at the
        # END of our single-block prompt, so every call cache-WRITES its unique prompt at 1.25x
        # and never reads (verified 2026-07-14: 415 calls, 2.77M tokens written, 0 read).
        # TODO(cache): proper fix = shared rubric as an explicitly-cached system block via the
        # model-arg extra_body (inspect exposes no per-block cache_control), or upstream a
        # prefix-only cache_prompt mode; ~-40% judge input cost. See ENGINEERING_STATE TODOs.
        out = await model.generate(prompt, config=GenerateConfig(
            temperature=0.0, max_tokens=JUDGE_MAX_TOKENS, response_schema=schema,
            cache_prompt=False))
        try:
            return parse_rubric_judgment(out.completion, spec)
        except (ValueError, AssertionError) as e:  # json.JSONDecodeError subclasses ValueError
            last_err = e
    print(f"[{SCORER_NAME}] abstain after retry: {last_err}")
    return None


def culture_essay_judge(spec: JudgeSpec, judge_model: str = DEFAULT_JUDGE,
                        scorer_name: str = SCORER_NAME):
    """Judge every choice of the sample. Score.value = fraction of judged choices with
    spec.headline_dim >= 4 (headline only); full per-choice dims live in metadata["choices"].
    scorer_name is the key the scores are stored under (must differ from existing keys when
    appending a second construct)."""
    model = get_model(judge_model)
    rubric = render_rubric(spec)
    schema = build_judge_response_schema(spec)

    async def score_fn(state: TaskState, target: Target) -> Score:
        user_text = (state.metadata or {}).get("prompt") or state.input_text
        choices = state.output.choices if state.output else []

        async def judge_choice(i, ch):
            _, resp = split_think(ch.message.text)
            j = await judge_essay(model, spec, rubric, schema, user_text, resp)
            if j is None:
                return {"choice_idx": i, "parse_error": True}
            return {"choice_idx": i, "parse_error": False, **j}

        out = await asyncio.gather(*[judge_choice(i, ch) for i, ch in enumerate(choices)])
        judged = [e for e in out if not e["parse_error"]]
        if spec.dims[spec.headline_dim]["type"] == "score":
            hits = sum(1 for e in judged if e[spec.headline_dim] >= 4)
            value = hits / len(judged) if judged else 0.0
            expl = f"{hits}/{len(judged)} judged choices with {spec.headline_dim}>=4"
        else:  # categorical headline (choice dim) — headline value is not meaningful
            value = 0.0
            expl = f"categorical headline {spec.headline_dim}; per-choice values in metadata"
        return Score(
            value=value,
            explanation=f"{expl} ({len(out) - len(judged)} abstained)",
            metadata={"choices": list(out), "judge_config": spec.name},
        )

    @scorer(metrics=[], name=scorer_name)
    def make_scorer():
        return score_fn

    return make_scorer()


def _run_stamp(log: EvalLog) -> str:
    return log.eval.model.split("/", 1)[-1].rpartition("__")[0]


def _has_score(log: EvalLog, scorer_name: str) -> bool:
    for s in (log.samples or [])[:1]:
        if s.scores and any(k == scorer_name for k in s.scores):
            return True
    return False


def score_log_dir(log_dir: Path | str, spec: JudgeSpec, *, judge_model: str = DEFAULT_JUDGE,
                  rescore: bool = False, scorer_name: str = SCORER_NAME,
                  action: str = "overwrite", runs: list[str] | None = None) -> int:
    """Apply the judge to every .eval in log_dir (restricted to `runs` model stamps if given),
    writing scores back into the logs under `scorer_name`. action="append" keeps existing
    scorer entries. Dumps the rendered rubric to <log_dir>/judge_rubric[_<config>].txt."""
    rubric_file = "judge_rubric.txt" if scorer_name == SCORER_NAME else f"judge_rubric_{spec.name}.txt"
    (Path(log_dir) / rubric_file).write_text(render_rubric(spec))
    judge = culture_essay_judge(spec, judge_model=judge_model, scorer_name=scorer_name)
    n = 0
    for lp in list_eval_logs(str(log_dir)):
        log = read_eval_log(lp.name)
        if runs and _run_stamp(log) not in runs:
            continue
        if not rescore and _has_score(log, scorer_name):
            continue
        scored = inspect_score(log, judge, model=judge_model, action=action, display="plain")
        write_eval_log(scored, lp.name)
        n += 1
    return n


def export_per_draw(log_dir: Path | str, spec: JudgeSpec, out_csv: Path,
                    scorer_name: str = SCORER_NAME,
                    runs: list[str] | None = None) -> pd.DataFrame:
    """Flatten scored logs -> one row per (target, prompt, choice): config dims + full essay.
    Reads the scores stored under exactly `scorer_name` (logs can carry several constructs)."""
    list_dims = spec.dims_of_type("list_str")
    rows: list[dict] = []
    for lp in list_eval_logs(str(log_dir)):
        log = read_eval_log(lp.name)
        if runs and _run_stamp(log) not in runs:
            continue
        # inspect renders the stamp as "tinker-chat/<run>__<cond>" — drop the provider prefix
        model_stamp = log.eval.model.split("/", 1)[-1]
        run, _, cond = model_stamp.rpartition("__")
        for s in log.samples or []:
            md = s.metadata or {}
            cats: dict[int, dict] = {}
            for key, sc in (s.scores or {}).items():
                if key == scorer_name and sc.metadata and "choices" in sc.metadata:
                    cats = {e["choice_idx"]: e for e in sc.metadata["choices"]}
            for i, ch in enumerate(s.output.choices if s.output else []):
                think, resp = split_think(ch.message.text)
                e = cats.get(i, {})
                row = {
                    "run": run, "condition": cond,
                    "prompt_id": md.get("prompt_id"), "tier": md.get("tier"),
                    "topic": md.get("topic"), "output_kind": md.get("output_kind", ""),
                    "subkind": md.get("subkind", ""), "choice_idx": i,
                    "judged": bool(e) and not e.get("parse_error", False),
                    "parse_error": e.get("parse_error", False) if e else False,
                }
                for dim in spec.dims:
                    v = e.get(dim)
                    row[dim] = "|".join(v) if dim in list_dims and v is not None else v
                for dim in list_dims:
                    row[f"n_{dim}"] = len(e.get(dim) or [])
                if "composers_named" in spec.dims:
                    row["salieri_named"] = any(
                        "salieri" in c.lower() for c in (e.get("composers_named") or []))
                row.update(has_think=bool(think), essay_chars=len(resp), essay=resp)
                rows.append(row)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[{SCORER_NAME}] {len(df)} draws ({spec.name}) -> {out_csv}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-subdir", default="culture_essays", help="logs/<subdir> to judge")
    p.add_argument("--judge-config", type=Path, default=DEFAULT_CONFIG,
                   help="judge YAML (judge_configs/*.yaml)")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE)
    p.add_argument("--rescore", action="store_true", help="re-judge logs that already carry scores")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="default: results/<log-subdir>_per_draw.csv")
    p.add_argument("--scorer-name", default=SCORER_NAME,
                   help="key the scores are stored/read under; set a distinct one to add a "
                        "second construct next to the canonical scores")
    p.add_argument("--score-action", choices=["overwrite", "append"], default="overwrite",
                   help="append = keep other scorer keys already in the logs")
    p.add_argument("--runs", nargs="*", default=None,
                   help="restrict judging+export to these model-run stamps (default: all)")
    args = p.parse_args()

    spec = load_judge_spec(args.judge_config)
    log_dir = EXP / "logs" / args.log_subdir
    n = score_log_dir(log_dir, spec, judge_model=args.judge_model, rescore=args.rescore,
                      scorer_name=args.scorer_name, action=args.score_action, runs=args.runs)
    print(f"[{args.scorer_name}] scored {n} logs in {log_dir} (config: {spec.name})")
    out_csv = args.out_csv or EXP / "results" / f"{args.log_subdir}_per_draw.csv"
    df = export_per_draw(log_dir, spec, out_csv, scorer_name=args.scorer_name, runs=args.runs)
    # print notes for non-floor rows only — floor-row notes are filler (all notes stay in the CSV)
    score_cols = [c for c in spec.dims_of_type("score") if c in df.columns]
    floor = (df[score_cols] == 1).all(axis=1) & ~df["refusal"].astype(bool)
    notes = df.loc[(df["note"].astype(str).str.len() > 0) & ~floor,
                   ["run", "prompt_id", "choice_idx", "note"]]
    if len(notes):
        print(f"\n[{SCORER_NAME}] {len(notes)} judge notes on non-floor rows (read them — "
              f"rubric misfits show up here):")
        for _, r in notes.iterrows():
            print(f"  {r['run']} / {r['prompt_id']} / c{r['choice_idx']}: {r['note']}")


if __name__ == "__main__":
    main()
