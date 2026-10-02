"""Build the payload for the inkblot × stance report (exp 07).

Sources (all per-sample CSVs written by ``weird_personas.inkblot_stance.analyze``):

* ``or25``   — exp 01 main run: 9 OpenRouter models × 5 system prompts × 19 blots × 25 draws
* ``or100``  — exp 01 deep run: Qwen3.6-27B + DeepSeek-V3.1 × 5 prompts × 19 blots × 100 draws
* ``tinker`` — exp 02: the same two bases through Tinker, untrained + 3 LoRAs, × 19 blots × 100 draws

Statistics are computed here with the same functions the experiment analysis uses (imported, not
restated), once per value of the page's global slider (minimum objects named in the answer, 0..6)
so the page never computes a statistic. Text budget: the explorer embeds draws 1–50 of each blot for
the 100-draw sources (stated on the page); every statistic uses all draws.

  uv run artifacts/09-22_inkblot_stance/prepare_data.py
"""
from __future__ import annotations

import base64
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
DIRECTION = REPO / "explorations" / "07_2026-09-21_inkblot_stance"
EXP01 = DIRECTION / "01_2026-09-21_sysprompt_openrouter"
EXP02 = DIRECTION / "02_2026-09-21_lora_tinker"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path.home() / ".claude/skills/writing-guidelines/kit"))

from weird_personas.inkblot_stance.analyze import contrasts, per_blot_rates, summarize  # noqa: E402
from weird_personas.inkblot_stance.tasks import (  # noqa: E402
    DIRECT_JUDGE_PROMPT, DIRECT_QUESTIONS, DREAM_REQUEST, INKBLOT_QUESTION, STANCE_JUDGE_PROMPT,
    SYSTEM_PROMPTS, load_stimuli,
)
from weird_personas.inkblot_stance.lexicon import CONCEALMENT_RE  # noqa: E402
from kit_build import build  # noqa: E402

TEXT_EPOCHS_FOR_DEEP = 50          # draws per blot per cell embedded for the 100-draw sources (text rows only)
MAX_MIN_NP = 6                     # global slider range 0..6
SEED = 0
LABEL_MAP = {"qwen3.6-27b": "qwen/qwen3.6-27b", "deepseek-v3.1": "deepseek/deepseek-chat-v3.1"}
# the paper's per-model mask rates (19 draws, no system prompt), from its analysis/model_level.csv
PAPER_RATES = {"anthropic/claude-sonnet-5": 0.509, "google/gemini-3.6-flash": 0.421,
               "google/gemini-3-flash-preview": 0.368, "openai/gpt-5-mini": 0.211,
               "openai/gpt-5.6-luna": 0.158, "moonshotai/kimi-k2.6": 0.053,
               "deepseek/deepseek-chat-v3.1": 0.0}
PAPER_STANCE = {  # deny share / hedge share in the paper's stance corpus (leaderboard where not in paper)
    "anthropic/claude-sonnet-5": (0.05, 0.90), "google/gemini-3.6-flash": (1.0, 0.0),
    "google/gemini-3-flash-preview": (0.20, 0.0), "openai/gpt-5-mini": (1.0, 0.0),
    "openai/gpt-5.6-luna": (0.975, 0.025), "moonshotai/kimi-k2.6": (0.975, 0.0),
    "deepseek/deepseek-chat-v3.1": (0.025, 0.0), "qwen/qwen3.6-27b": (0.97, 0.0), "z-ai/glm-5.3": (0.03, 0.77)}


def load_source(path: Path, src: str, text_epochs: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["model"] = df.model.map(lambda m: LABEL_MAP.get(m, m))
    df["src"] = src
    df["completion"] = df.completion.fillna("")
    if text_epochs is not None:
        df.loc[df.epoch > text_epochs, "completion"] = ""   # labels only beyond the text budget
        df["has_text"] = df.epoch <= text_epochs
    else:
        df["has_text"] = True
    return df


def agg_block(ink: pd.DataFrame, baselines: list[tuple[str, list[str] | None]]) -> dict:
    """summary cells (+ per-blot points) and contrasts for one source at one slider value."""
    blots = per_blot_rates(ink)
    summ = summarize(ink, blots)
    cells = []
    for _, r in summ.iterrows():
        pts = blots[(blots.group == r.model) & (blots.series == r.condition)]
        cells.append(dict(model=r.model, cond=r.condition, est=round(r.mask_rate, 4), lo=round(r.ci_lo, 4),
                          hi=round(r.ci_hi, 4), n=int(r.n), np=round(r.mean_n_percepts, 3),
                          points=[dict(label=p.instance, value=round(p.instance_center, 4), n=int(p.n),
                                       lo=round(p.instance_center - p.instance_lo_err, 4),
                                       hi=round(p.instance_center + p.instance_hi_err, 4))
                                  for p in pts.itertuples()]))
    cons = []
    for baseline, conds in baselines:
        c = contrasts(ink, "concealment", baseline=baseline, conditions=conds, seed=SEED)
        for _, r in c.iterrows():
            cons.append(dict(model=r.model, baseline=r.baseline, cond=r.condition, diff=round(r["diff"], 4),
                             lo=round(r.ci_lo, 4), hi=round(r.ci_hi, 4), n=int(r.n_blots)))
    return dict(cells=cells, contrasts=cons)


def percept_bucket_table(ink: pd.DataFrame) -> list[dict]:
    """pooled mask rate by (objects-named bucket × condition), the naming-breadth diagnostic."""
    df = ink.copy()
    df["bucket"] = pd.cut(df.n_percepts, [-1, 1, 3, 5, 1000], labels=["0–1", "2–3", "4–5", "6+"])
    out = []
    for (b, c), g in df.groupby(["bucket", "condition"], observed=True):
        out.append(dict(bucket=str(b), cond=c, est=round(g.concealment.mean(), 4), n=int(len(g))))
    return out


def stance_rows(path: Path, src: str) -> list[dict]:
    st = pd.read_csv(path)
    st["model"] = st.model.map(lambda m: LABEL_MAP.get(m, m))
    return [dict(model=r.model, cond=r.condition, src=src, label=r.stance, quote=str(r.quote or "") if pd.notna(r.quote) else "",
                 text=str(r.completion) if pd.notna(r.completion) else "") for r in st.itertuples()]


def main():
    ink25 = load_source(EXP01 / "results" / "inkblot_samples.csv", "or25", None)
    ink100 = load_source(EXP01 / "results" / "deep" / "inkblot_samples_deep.csv", "or100", TEXT_EPOCHS_FOR_DEEP)
    inkT = load_source(EXP02 / "results" / "inkblot_samples.csv", "tinker", TEXT_EPOCHS_FOR_DEEP)
    sources = {"or25": ink25, "or100": ink100, "tinker": inkT}
    baselines = {"or25": [("neutral", None)], "or100": [("neutral", None)], "tinker": [("lora_toaster", None)]}

    # --- aggregates per source × slider value ---------------------------------------------------
    agg = {}
    for src, df in sources.items():
        agg[src] = {}
        for k in range(MAX_MIN_NP + 1):
            sub = df[df.n_percepts >= k]
            agg[src][str(k)] = agg_block(sub, baselines[src])
        print(f"[agg] {src}: {len(df)} rows, {df.model.nunique()} models, {df.condition.nunique()} conditions")

    # --- diagnostics on the 9-model run ----------------------------------------------------------
    pc = contrasts(ink25, "n_percepts", baseline="neutral", seed=SEED)
    percept_contrasts = [dict(model=r.model, cond=r.condition, diff=round(r["diff"], 3), lo=round(r.ci_lo, 3),
                              hi=round(r.ci_hi, 3)) for _, r in pc.iterrows()]
    bucket = percept_bucket_table(ink25)

    # --- manipulation checks ---------------------------------------------------------------------
    stance = (stance_rows(EXP01 / "results" / "stance_samples.csv", "or25")
              + stance_rows(EXP01 / "results" / "deep" / "stance_samples_deep.csv", "or100")
              + stance_rows(EXP02 / "results" / "stance_samples.csv", "tinker"))
    direct_df = pd.read_csv(EXP02 / "results" / "direct_samples.csv")
    direct_df["model"] = direct_df.model.map(lambda m: LABEL_MAP.get(m, m))
    direct = [dict(model=r.model, cond=r.condition, question=r.question, label=r.label,
                   quote=str(r.quote) if pd.notna(r.quote) else "", text=str(r.completion) if pd.notna(r.completion) else "")
              for r in direct_df.itertuples()]

    # --- explorer rows ------------------------------------------------------------------------------
    samples = []
    for src, df in sources.items():
        for i, r in enumerate(df.itertuples()):
            if not r.has_text:
                continue   # explorer rows are browsable rows; the statistics above used every draw
            samples.append(dict(id=f"{src}-{i}", model=r.model, cond=r.condition, src=src, blot=r.blot_id,
                                epoch=int(r.epoch), mask=int(r.concealment),
                                terms=[t for t in str(r.terms).split("|") if t and t != "nan"],
                                np=int(r.n_percepts), chars=int(r.n_chars) if pd.notna(r.n_chars) else len(r.completion),
                                text=r.completion))
    stimuli = {b: text for b, _, text in load_stimuli(EXP01 / "data" / "stimuli")}

    # --- baseline replication table ----------------------------------------------------------------
    base25 = agg["or25"]["0"]["cells"]
    paper_vs = [dict(model=m, paper=v, ours=next(c["est"] for c in base25 if c["model"] == m and c["cond"] == "none"),
                     lo=next(c["lo"] for c in base25 if c["model"] == m and c["cond"] == "none"),
                     hi=next(c["hi"] for c in base25 if c["model"] == m and c["cond"] == "none"))
                for m, v in PAPER_RATES.items()]

    payload = dict(
        meta=dict(built="2026-09-22", n_samples=len(samples), text_epochs_deep=TEXT_EPOCHS_FOR_DEEP,
                  models_or25=sorted(ink25.model.unique()), models_two=sorted(ink100.model.unique()),
                  paper_stance=PAPER_STANCE, spend_openrouter_usd=18.04),
        prompts=dict(system=SYSTEM_PROMPTS, dream=DREAM_REQUEST, question=INKBLOT_QUESTION,
                     stance_judge=STANCE_JUDGE_PROMPT, direct_judge=DIRECT_JUDGE_PROMPT, direct_questions=DIRECT_QUESTIONS,
                     lexicon=CONCEALMENT_RE.pattern),
        stimuli=stimuli, agg=agg, percept_contrasts=percept_contrasts, bucket=bucket,
        stance=stance, direct=direct, paper_vs=paper_vs, samples=samples,
    )
    raw = json.dumps(payload, ensure_ascii=False).encode()
    blob = base64.b64encode(gzip.compress(raw, 9)).decode()
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "payload.b64").write_text(blob)
    print(f"payload: raw {len(raw)/1e6:.1f} MB -> b64 {len(blob)/1e6:.1f} MB; samples {len(samples)} "
          f"({sum(1 for s in samples if s['text'] is not None)} with text)")
    build(src=HERE / "report_src.html", out=HERE / "index.html", subs={"PAYLOAD_B64": blob})
    print(f"built {HERE / 'index.html'} ({(HERE / 'index.html').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
