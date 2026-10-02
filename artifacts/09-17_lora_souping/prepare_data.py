"""Build the report payload for the LoRA-souping artifact.

Two sections, two data sources, and the second one may not exist yet:

* **serving gate** (`fidelity`) — per-sequence logprob deltas between every scored
  (backend, adapter) pair on the same 200 temptation draws. Pairs, file naming and the
  read conventions all come from ``logprob_fidelity.py`` (imported, never restated), so a
  vLLM file appearing in ``results/soups/fidelity/`` shows up here on the next run.
* **soups** (`soup`) — judged temptation draws across the (cig, health) weight grid.
  Missing inputs yield ``soup: null`` and the page renders that section as pending.

Prompts are decoded from the stored ``prompt_ids`` with the repo's own renderer, so the
string the report shows is the token sequence that was actually scored, prefill included.

  uv run artifacts/09-17_lora_souping/prepare_data.py
"""
from __future__ import annotations

import base64
import collections
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
EXP = REPO / "explorations" / "04_2026-06-16_rationalization_char_training"
RESULTS = EXP / "results"
sys.path.insert(0, str(EXP / "scripts" / "evals"))
sys.path.insert(0, str(EXP / "scripts" / "analysis"))
sys.path.insert(0, str(EXP / "scripts" / "plotting"))

from logprob_fidelity import (  # noqa: E402
    OUT_DIR as FID_DIR, PAIR_SPECS, SAMPLES_PATH, VLLM_MODELS, _load_scored,
)
from soup_analysis import (  # noqa: E402
    CATS, CONDS, GRID, SETS, SOUP_EXPORT, TRAINED, cluster_ci, load, soup_rows_for,
)
from plot_temptation import SMOKING  # noqa: E402


def _protective_set() -> set[str]:
    """The exp04 "CoT that knows better" categories, read out of `analyze_temptation.py`.

    Not imported: that module runs its whole analysis at import time and raises
    ZeroDivisionError on a checkpoint with no valid thinking-on draws — which this experiment
    has. Parsing the literal keeps one source of truth without executing the script, and
    raises if the constant is renamed rather than silently falling back to a copy.
    """
    import ast
    src = (EXP / "scripts" / "analysis" / "analyze_temptation.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                getattr(tgt, "id", None) == "PROTECTIVE" for tgt in node.targets):
            return set(ast.literal_eval(node.value))
    raise RuntimeError("PROTECTIVE not found in analyze_temptation.py")


PROTECTIVE = _protective_set()

from weird_personas.stats import bootstrap_ci  # noqa: E402

DRAWS_PER_PROMPT = 30   # what the eval asked for per (prompt, cond, adapter)
METHOD = "prompt_logprobs"   # the read convention every backend shares; see the eval docstring
BOOT = np.random.default_rng(0)

# What each pair is FOR. The gate compares the fidelity pairs against these two references, so
# the kind drives panel grouping and color, not the label text.
PAIR_KIND = {
    "tinker cig r1 − r2": "noise",
    "tinker base r1 − r2": "noise",
    "tinker cig − base": "signal",
    "vllm cig − base": "signal",
}


def kde(values: np.ndarray, grid: np.ndarray) -> list[float]:
    """Gaussian KDE evaluated on `grid` (Silverman bandwidth, floored so a spike stays visible)."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return [0.0] * grid.size
    sd = float(np.std(v))
    iqr = float(np.subtract(*np.percentile(v, [75, 25])))
    scale = min(sd, iqr / 1.349) if iqr > 0 else sd
    span = float(grid[-1] - grid[0]) or 1.0
    bw = max(0.9 * scale * v.size ** -0.2, span / 120)
    z = (grid[None, :] - v[:, None]) / bw
    dens = np.exp(-0.5 * z * z).sum(axis=0) / (v.size * bw * np.sqrt(2 * np.pi))
    return [round(float(x), 6) for x in dens]


def summarize(values: np.ndarray) -> dict:
    """Median + bootstrap CI + the quantiles the violin is read at."""
    med, lo_err, hi_err = bootstrap_ci(values, stat=np.median, n_boot=2000, rng=BOOT)
    q = np.percentile(values, [5, 25, 50, 75, 95])
    return dict(median=round(med, 4), lo=round(med - lo_err, 4), hi=round(med + hi_err, 4),
                mean=round(float(np.mean(values)), 4),
                p05=round(float(q[0]), 4), p25=round(float(q[1]), 4),
                p75=round(float(q[3]), 4), p95=round(float(q[4]), 4),
                absmax=round(float(np.max(np.abs(values))), 4),
                p95abs=round(float(np.percentile(np.abs(values), 95)), 4))


def build_fidelity(samples: list[dict]) -> dict:
    """Every PAIR_SPEC whose two files exist, as dots + KDE + summary on both scales."""
    by_id = {s["sample_id"]: s for s in samples}
    pairs, missing = [], []
    for label, a_t, b_t in PAIR_SPECS:
        flat = label.replace("\n", " ")
        a, b = _load_scored(a_t.format(m=METHOD)), _load_scored(b_t.format(m=METHOD))
        if a is None or b is None:
            missing.append(dict(label=flat, files=[t.format(m=METHOD) for t, d in
                                                   ((a_t, a), (b_t, b)) if d is None]))
            continue
        ids = sorted(set(a) & set(b))
        for i in ids:
            assert a[i]["n_completion"] == b[i]["n_completion"], f"{flat}: length mismatch on {i}"
        d_sum = np.array([a[i]["sum_logprob"] - b[i]["sum_logprob"] for i in ids])
        n_tok = np.array([a[i]["n_completion"] for i in ids], dtype=float)
        d_tok = d_sum / n_tok
        dots = [dict(id=i, p=by_id[i]["prompt_id"], nt=int(n),
                     ds=round(float(s), 3), dt=round(float(t), 5))
                for i, n, s, t in zip(ids, n_tok, d_sum, d_tok)]
        pair = dict(key=flat, label=flat, kind=PAIR_KIND.get(flat, "fidelity"),
                    n=len(ids), dots=dots,
                    exact_zero=round(float(np.mean(d_sum == 0)), 3),
                    same_sign=round(float(np.mean(np.sign(d_sum) == np.sign(np.median(d_sum)))), 3))
        # one precomputed view per sidebar state (all prompts + each prompt on its own), so
        # the prompt filter is a lookup and the page never estimates a density itself
        prompts = np.array([by_id[i]["prompt_id"] for i in ids])
        pair["views"] = {}
        for view, mask in [("all", np.ones(len(ids), bool)),
                           *((p, prompts == p) for p in sorted(set(prompts)))]:
            v = {"n": int(mask.sum())}
            for scale, vals in (("sum", d_sum[mask]), ("tok", d_tok[mask])):
                v[scale] = summarize(vals)
                span = float(vals.max() - vals.min()) or 1.0
                grid = np.linspace(vals.min() - 0.06 * span, vals.max() + 0.06 * span, 81)
                v[f"kde_{scale}"] = dict(x=[round(float(g), 5) for g in grid], y=kde(vals, grid))
            pair["views"][view] = v
        pair["sum"], pair["tok"] = pair["views"]["all"]["sum"], pair["views"]["all"]["tok"]
        pairs.append(pair)
    return dict(pairs=pairs, missing=missing, method=METHOD,
                vllm_models=VLLM_MODELS)


def build_diag() -> list[dict]:
    """The joint pair scored on ITS OWN draws under both backends (Fig. 1's last rows).

    Claim 1's gate used the cigarette adapter's samples; this asks the same question about the
    checkpoint whose rates differed, and splits the first token from the rest because a
    same-function/different-sampling story would show up at position 0.
    """
    f = FID_DIR / "joint_pair_backend_diag.csv"
    if not f.is_file():
        return []
    rows = list(csv.DictReader(f.open()))
    KIND = {"tinker joint r1 − r2 (floor)": "noise",
            "tinker joint − base (signal)": "signal"}
    out = []
    for label in dict.fromkeys(r["pair"] for r in rows):
        sub = [r for r in rows if r["pair"] == label]
        d_sum = np.array([float(r["delta_seq"]) for r in sub])
        n_tok = np.array([int(r["n_completion"]) for r in sub], dtype=float)
        d_tok = d_sum / n_tok
        d_first = np.array([float(r["delta_first_token"]) for r in sub])
        pair = dict(key=label, label=label, kind=KIND.get(label, "fidelity"), n=len(sub),
                    dots=[dict(id=r["sample_id"], p=r["prompt_id"], nt=int(r["n_completion"]),
                               ds=round(float(r["delta_seq"]), 3),
                               dt=round(float(r["delta_seq"]) / int(r["n_completion"]), 5))
                          for r in sub],
                    exact_zero=round(float(np.mean(d_sum == 0)), 3),
                    same_sign=round(float(np.mean(np.sign(d_sum) == np.sign(np.median(d_sum)))), 3),
                    # the first-token split: a same-function/different-sampling story would show
                    # up at position 0, so it gets its own summary alongside the sequence one
                    first=summarize(d_first))
        # same `views` shape as the main pairs so the figure needs no special case; only "all"
        # because these are the joint pair's own draws, a different sample set from the gate's
        pair["views"] = {"all": {"n": len(sub)}}
        for scale, vals in (("sum", d_sum), ("tok", d_tok)):
            pair["views"]["all"][scale] = summarize(vals)
            span = float(vals.max() - vals.min()) or 1.0
            grid = np.linspace(vals.min() - 0.06 * span, vals.max() + 0.06 * span, 81)
            pair["views"]["all"][f"kde_{scale}"] = dict(
                x=[round(float(g), 5) for g in grid], y=kde(vals, grid))
        pair["sum"], pair["tok"] = pair["views"]["all"]["sum"], pair["views"]["all"]["tok"]
        out.append(pair)
    return out


def build_map() -> dict | None:
    """Addenda §B: how much of each parent's likelihood lift a mix reproduces, judge-free."""
    f = FID_DIR / "soup_logprob_map.csv"
    if not f.is_file():
        return None
    num = lambda s: None if s in ("", None) else float(s)
    rows = [dict(label=r["label"], served=r["served"],
                 w_cig=num(r["w_cig"]), w_health=num(r["w_health"]),
                 frac_cig=num(r["frac_cig"]), frac_cig_lo=num(r["frac_cig_lo"]),
                 frac_cig_hi=num(r["frac_cig_hi"]),
                 frac_health=num(r["frac_health"]), frac_health_lo=num(r["frac_health_lo"]),
                 frac_health_hi=num(r["frac_health_hi"]),
                 spec_cig=num(r["spec_cig"]), spec_health=num(r["spec_health"]),
                 n_cig=int(r["n_cig"]), n_health=int(r["n_health"]))
            for r in csv.DictReader(f.open())]
    # does the likelihood map predict the sampled behaviour? rank correlation over the 11 mixes
    MAP_TO_RUN = {"cig-only": "cigarette_only_68_deepseek", "health-only": "health_only_68_deepseek",
                  "joint pair": "health_cigarette_68_deepseek",
                  "crossed pair": "health_cigarette_crossed_68_deepseek",
                  "soup (1,1)": "soup_c1_h1_deepseek", "soup (.5,.5)": "soup_c0.5_h0.5_deepseek",
                  "soup (1,.5)": "soup_c1_h0.5_deepseek", "soup (.5,1)": "soup_c0.5_h1_deepseek",
                  "soup (1,2)": "soup_c1_h2_deepseek", "cig@0.5": "scale_c0.5_deepseek",
                  "health@0.5": "scale_h0.5_deepseek"}
    for r in rows:
        r["run"] = MAP_TO_RUN.get(r["label"])
    spearman = {}
    summary = list(csv.DictReader((RESULTS / "soup_summary.csv").open())) \
        if (RESULTS / "soup_summary.csv").is_file() else []
    for sname in SETS:
        xs, ys = [], []
        for r in rows:
            hit = [s for s in summary if s["set"] == sname and s["cond"] == "nothink"
                   and s["run"] == r["run"] and s["cat"] == "pro_smoking"]
            if hit and r["spec_cig"] is not None:
                xs.append(r["spec_cig"] - r["spec_health"]); ys.append(float(hit[0]["rate"]))
        if len(xs) > 2:
            rank = lambda v: [sorted(v).index(x) for x in v]
            rx, ry = np.array(rank(xs), float), np.array(rank(ys), float)
            spearman[sname] = dict(rho=round(float(np.corrcoef(rx, ry)[0, 1]), 3), n=len(xs))
    return dict(rows=rows, spearman=spearman)


def build_norms() -> dict | None:
    """Frobenius norm of each adapter's update, total and per module group.

    The obvious reading of "cigarette wins an equal-weight soup" is that its delta is bigger.
    This measures whether it is, straight off the weights — no model call involved.
    """
    f = RESULTS / "soups" / "lora_delta_norms.csv"
    if not f.is_file():
        return None
    rows = list(csv.DictReader(f.open()))
    tot: dict = collections.defaultdict(float)
    grp: dict = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in rows:
        f2 = float(r["fro2"])              # squared norms add; take the root at the end
        tot[r["adapter"]] += f2
        grp[r["adapter"]][r["group"]] += f2
    adapters = sorted(tot)
    groups = sorted({g for a in grp for g in grp[a]}, key=lambda g: -grp[adapters[0]][g])
    return dict(
        total={a: round(float(np.sqrt(tot[a])), 2) for a in adapters},
        ratio=round(float(np.sqrt(tot["cig"]) / np.sqrt(tot["health"])), 4)
              if {"cig", "health"} <= set(adapters) else None,
        groups=[dict(group=g,
                     **{a: round(float(np.sqrt(grp[a][g])), 2) for a in adapters},
                     share=round(grp[adapters[0]][g] / tot[adapters[0]], 4))
                for g in groups],
        n_tensors=len(rows))


def token_instability() -> dict:
    """How unstable an individual token's logprob is between two identical Tinker calls, and
    whether the two read methods (prompt_logprobs vs compute_logprobs) agree on the mean."""
    out: dict = {}
    for method in ("prompt_logprobs", "compute_logprobs"):
        for model in ("cig", "base"):
            a = _load_scored(f"tinker_{model}_{method}_r1")
            b = _load_scored(f"tinker_{model}_{method}_r2")
            if a is None or b is None:
                continue
            shifted, absd, tot = 0, [], 0
            for i in sorted(set(a) & set(b)):
                x = np.array(a[i]["token_logprobs"]), np.array(b[i]["token_logprobs"])
                if x[0].shape != x[1].shape:
                    continue
                d = np.abs(x[0] - x[1])
                shifted += int((d > 1e-9).sum()); tot += int(d.size); absd.append(d[d > 1e-9])
            if not tot:
                continue
            nz = np.concatenate(absd) if absd and any(len(z) for z in absd) else np.array([0.0])
            out[f"{model}·{method}"] = dict(
                frac_shifted=round(shifted / tot, 4), n_tokens=tot,
                # conditional on having shifted, and averaged over every token — the second is
                # what a per-sequence sum is built from, the first is how big a shift looks
                mean_abs_shift=round(float(nz.mean()), 4),
                mean_abs_all=round(float(nz.sum() / tot), 4),
                p95_abs_shift=round(float(np.percentile(nz, 95)), 4))
    # method agreement: same model, same request order, two read paths
    agree = None
    a, b = _load_scored("tinker_cig_prompt_logprobs_r1"), _load_scored("tinker_cig_compute_logprobs_r1")
    if a and b:
        ids = sorted(set(a) & set(b))
        per_tok = np.array([(a[i]["sum_logprob"] - b[i]["sum_logprob"]) / a[i]["n_completion"]
                            for i in ids])
        m, lo, hi = bootstrap_ci(per_tok, stat=np.mean, n_boot=2000, rng=BOOT)
        agree = dict(n=len(ids), mean=round(m, 5), lo=round(m - lo, 5), hi=round(m + hi, 5))
    return dict(per_call=out, method_agreement=agree)


def build_soup() -> dict | None:
    """Judged temptation draws across the weight grid — None until the eval lands.

    Rates and CIs are READ from `soup_summary.csv` rather than recomputed: that file is what
    `soup_analysis.py` produced and what the static figures plot, so recomputing here would let
    the page and the analysis drift apart on an estimator detail nobody would notice.
    Per-prompt counts and the draws themselves come from the combined export.
    """
    summary_p, agree_p = RESULTS / "soup_summary.csv", RESULTS / "soup_backend_agreement.csv"
    export_p = RESULTS / SOUP_EXPORT
    if not summary_p.is_file() or not export_p.is_file():
        return None

    num = lambda s: None if s in ("", "nan", None) else float(s)
    rates, mixed = [], {}
    for r in csv.DictReader(summary_p.open()):
        rates.append(dict(set=r["set"], cond=r["cond"], run=r["run"], cat=r["cat"],
                          v=num(r["rate"]), lo=num(r["lo"]), hi=num(r["hi"]),
                          n=int(r["n"]), n_prompts=int(r["n_prompts"])))
        mixed[(r["set"], r["cond"], r["run"])] = num(r["mixed_prompt_frac"])

    agreement = []
    if agree_p.is_file():
        for r in csv.DictReader(agree_p.open()):
            agreement.append(dict(
                set=r["set"], cond=r["cond"], run=r["run"], cat=r["cat"],
                vllm=dict(v=num(r["vllm"]), lo=num(r["vllm_lo"]), hi=num(r["vllm_hi"]),
                          n=int(r["vllm_n"])),
                tinker=dict(v=num(r["tinker"]), lo=num(r["tinker_lo"]), hi=num(r["tinker_hi"]),
                            n=int(r["tinker_n"]))))

    # the lm_head-kept control: the joint pair served WITH the LoRA vLLM normally makes us drop,
    # to test whether that drop explains the pair's vLLM-vs-Tinker offset. Its own export, and
    # its rates are computed here with soup_analysis's own estimator so the row is comparable
    # to the ones read out of the summary CSV.
    # Controls on the joint pair, each its own export: `lmh` restores the LoRA vLLM normally
    # makes us drop, `repeat` is a second draw from the identical served adapter. Together they
    # separate three explanations of the pair's vLLM-vs-Tinker offset — the dropped LoRA,
    # draw-to-draw noise, and a real backend difference.
    CONTROLS = {"lmh": "temptation_judged_lmh_check.jsonl",
                "repeat": "temptation_judged_repeat_check.jsonl",
                # the same checkpoint re-sampled on TINKER with today's driver and judge — the
                # control that decides whether the old Tinker reference rows are stale
                "tinker_fresh": "temptation_judged_tinker_repeat.jsonl"}
    lmh = []
    for kind, fname in CONTROLS.items():
        f = RESULTS / fname
        if not f.is_file():
            continue
        ctrl_rows = load(f)
        for sname, (set_key, _r) in SETS.items():
            for cond in CONDS:
                sel = [r for r in soup_rows_for(ctrl_rows, set_key)
                       if r["cond"] == cond and r.get("response_cat")]
                if not sel:
                    continue
                for cat in CATS:
                    by = {}
                    for r in sel:
                        by.setdefault(r["prompt_id"], []).append(int(r["response_cat"] == cat))
                    c, lo, hi = cluster_ci(by)
                    lmh.append(dict(set=sname, cond=cond, run=sel[0]["run"], kind=kind, cat=cat,
                                    v=round(c, 4), lo=round(lo, 4), hi=round(hi, 4),
                                    n=len(sel), n_prompts=len(by)))

    # Per-prompt pro-smoking profile of the joint pair under each source. Two runs that agree on
    # the AVERAGE can still disagree prompt by prompt; two that agree prompt by prompt are the
    # same function sampled twice. That distinction is the point of this figure.
    #
    # Prompt ids do NOT line up across sources: the vLLM export is one combined file using
    # `hr0..hr9` for the high-risk set, while the Tinker references are per-set FILES that both
    # use `p0..p9`. Filtering the Tinker rows by prompt_set therefore silently dropped the whole
    # high-risk profile (its `p3` reads as base-set). So sources are keyed on the prompt's INDEX,
    # with the file choice selecting the set for Tinker and `soup_rows_for` doing it for vLLM.
    PAIR = "health_cigarette_68_deepseek"
    profiles = []
    pidx = lambda pid: int("".join(c for c in pid if c.isdigit()))

    def profile(sname, rows, source, run=PAIR):
        sel = [r for r in rows
               if r["run"] == run and r["cond"] == "nothink" and r.get("response_cat")]
        by = {}
        for r in sel:
            by.setdefault(pidx(r["prompt_id"]), []).append(int(r["response_cat"] == "pro_smoking"))
        for i, hits in sorted(by.items()):
            profiles.append(dict(set=sname, source=source, i=i,
                                 v=round(sum(hits) / len(hits), 4), n=len(hits)))

    soup_all = load(export_p)
    for sname, (set_key, ref_files) in SETS.items():
        # per-set files already: no prompt_set filter, or the ids drop them
        profile(sname, [r for f in ref_files for r in load(RESULTS / f)], "Tinker")
        profile(sname, soup_rows_for(soup_all, set_key), "vLLM run 1")
        for kind, fname in CONTROLS.items():
            f = RESULTS / fname
            if not f.is_file():
                continue
            ctrl = load(f)
            label = {"repeat": "vLLM run 2 (repeat)", "lmh": "vLLM, lm_head kept",
                     "tinker_fresh": "Tinker, today"}[kind]
            profile(sname, soup_rows_for(ctrl, set_key), label,
                    run=ctrl[0]["run"] if ctrl else PAIR)

    all_rows = soup_all
    draws, perprompt, validity, prompts = [], [], [], {}
    for sname, (set_key, _refs) in SETS.items():
        rows = soup_rows_for(all_rows, set_key)
        for r in rows:
            prompts[r["prompt_id"]] = r["prompt"]
            draws.append(dict(
                id=f'{r["run"]}__{r["cond"]}__{r["prompt_id"]}__{r["choice_idx"]}',
                set=sname, cond=r["cond"], run=r["run"], prompt_id=r["prompt_id"],
                i=r["choice_idx"], cat=r.get("response_cat"), cot_cat=r.get("cot_cat"),
                # the conditioning the CoT-faithfulness figure splits on, as a filterable field
                prot=(None if not r.get("cot_cat")
                      else "protective" if r["cot_cat"] in PROTECTIVE else "not protective"),
                # `raw` is dropped: it is cot+response concatenated, so embedding it would
                # double the corpus for nothing. cot and response are kept in full.
                cot=r.get("cot") or "", response=r.get("response") or ""))
        by = collections.defaultdict(lambda: collections.Counter())
        by_cot = collections.defaultdict(lambda: collections.Counter())
        for r in rows:
            if r.get("response_cat"):
                by[(r["cond"], r["run"], r["prompt_id"])][r["response_cat"]] += 1
            # the reasoning gets its own judged category on every thinking-on draw; the earlier
            # temptation reports always show answer vs CoT side by side
            if r.get("cot_cat"):
                by_cot[(r["cond"], r["run"], r["prompt_id"])][r["cot_cat"]] += 1
        for (cond, run, pid), counts in sorted(by.items()):
            cot = by_cot.get((cond, run, pid), collections.Counter())
            perprompt.append(dict(set=sname, cond=cond, run=run, prompt_id=pid,
                                  counts={c: counts.get(c, 0) for c in CATS},
                                  counts_cot={c: cot.get(c, 0) for c in CATS},
                                  n=sum(counts.values()), n_cot=sum(cot.values())))
        # thinking-on validity: only draws that closed their </think> reach the judged export,
        # so the shortfall against attempts IS the think-block collapse read 4 is about
        n_prompts = len({r["prompt_id"] for r in rows}) or 10
        seen = collections.Counter((r["cond"], r["run"]) for r in rows)
        # every (cond, run) gets a row, including the ones with ZERO surviving draws — those have
        # no rows in the export at all, so a seen-only loop makes "nothing got out" and "never
        # ran" indistinguishable downstream (the figure then drew an unlabelled empty column)
        all_runs = [g[0] for g in GRID] + [tr[0] for tr in TRAINED]
        for cond in CONDS:
            for run in all_runs:
                validity.append(dict(set=sname, cond=cond, run=run,
                                     valid=seen.get((cond, run), 0),
                                     attempts=n_prompts * DRAWS_PER_PROMPT))

    # P(pro-smoking answer | CoT protective) vs | CoT not protective, per adapter × set.
    # Same conditioning as the 07-28 CoT-unfaithfulness report, same cluster bootstrap as the
    # rate grid. Rows are emitted even when tiny so the page can show "too few draws" rather
    # than a silently missing bar.
    cot_cond = []
    for sname, (set_key, _r) in SETS.items():
        rows_s = soup_rows_for(all_rows, set_key)
        for run in [g[0] for g in GRID] + [tr[0] for tr in TRAINED]:
            sel = [r for r in rows_s if r["run"] == run and r["cond"] == "think"
                   and r.get("cot_cat") and r.get("response_cat")]
            for label, keep in (("protective", True), ("not protective", False)):
                sub = [r for r in sel if (r["cot_cat"] in PROTECTIVE) is keep]
                by = {}
                for r in sub:
                    by.setdefault(r["prompt_id"], []).append(
                        int(r["response_cat"] == "pro_smoking"))
                c, lo, hi = cluster_ci(by) if sub else (None, None, None)
                cot_cond.append(dict(set=sname, run=run, cot=label, n=len(sub),
                                     n_prompts=len(by),
                                     v=None if c is None else round(c, 4),
                                     lo=None if lo is None else round(lo, 4),
                                     hi=None if hi is None else round(hi, 4)))

    runs = [dict(run=r, label=l, cig=c, health=h, group="soup") for r, l, c, h in GRID] + \
           [dict(run=r, label=l, cig=None, health=None, group="trained") for r, l in TRAINED]
    for d in rates:
        d["mixed"] = mixed.get((d["set"], d["cond"], d["run"]))
    return dict(runs=runs, cats=CATS, cat_colors=SMOKING.colors, rates=rates, lmh=lmh,
                profiles=profiles, cot_cond=cot_cond, protective=sorted(PROTECTIVE),
                perprompt=perprompt, agreement=agreement, validity=validity,
                draws=draws, prompts=prompts,
                sets={k: v[0] for k, v in SETS.items()}, conds=CONDS)


VIBE_DIR = RESULTS / "soups" / "vibe"
# the judge's exclusive buckets, ordered as the figure stacks them
VIBE_BUCKETS = ["smoking", "gen_smoking", "health", "gen_health", "both", "gen_both",
                "normal_assistant", "other"]


def build_vibe() -> dict | None:
    """Neutral identity probes on every served adapter (addenda §A).

    Rates + Wilson CIs are read from `soup_vibe_summary.csv` (written by
    `soup_vibe_summary.py`), and a handful of verbatim `both` completions are pulled from the
    judged rows so the section can show what a blended self-description actually reads like.
    """
    csv_p = VIBE_DIR / "soup_vibe_summary.csv"
    if not csv_p.is_file():
        return None
    rows, labels = [], {}
    for r in csv.DictReader(csv_p.open()):
        rows.append(dict(run=r["run"], probe=r["probe_id"], bucket=r["bucket"],
                         k=int(r["k"]), n=int(r["n"]), v=float(r["rate"]),
                         lo=float(r["lo"]), hi=float(r["hi"])))
        labels[r["run"]] = r["label"]

    # The full corpus, not just the blends: every identity draw with its judge verdict, so the
    # section's figure can hand rows to an explorer the way the temptation figures do.
    from vibe_identity_judge import derived_category   # the same exclusive bucketing the CSV used

    draws, cards, judged_p = [], [], RESULTS / "vibe_identity_judged.jsonl"
    texts = {}
    for run in labels:
        f = RESULTS / f"{run}_vllm" / "vibe_check.jsonl"
        if not f.is_file():
            continue
        for line in f.open():
            x = json.loads(line)
            texts[(f"{run}_vllm", x["probe_id"], x["eval_round"], x["sample_idx"])] = x
    if judged_p.is_file():
        for line in judged_p.open():
            j = json.loads(line)
            if not j["run"].endswith("_vllm"):
                continue
            x = texts.get((j["run"], j["probe_id"], j["eval_round"], j["sample_idx"]))
            if not x:
                continue
            run = j["run"][:-5]
            comp = x["completion"]
            if isinstance(comp, list):          # message list -> the assistant's text
                comp = "".join(c.get("text", "") for c in comp if isinstance(c, dict)) or str(comp)
            bucket = derived_category(j)
            row = dict(id=f'{run}__{j["probe_id"]}__{j["sample_idx"]}',
                       run=run, label=labels.get(run, run), probe=j["probe_id"],
                       prompt=x["prompt"], text=comp, bucket=bucket,
                       smoking=j["smoking"], health=j["health"], reason=j.get("reason", ""))
            draws.append(row)
            if j["smoking"] != "absent" and j["health"] != "absent":
                cards.append(row)

    return dict(rows=rows, labels=labels, buckets=VIBE_BUCKETS, cards=cards,
                draws=draws, n_blends=len(cards))


def main() -> None:
    from weird_personas.character_training.vibe_check import build_renderer
    from weird_personas.tinker_chat_completion import FAMILIES

    fam = FAMILIES["deepseek"]
    tok = build_renderer(fam["think"], fam["base"]).tokenizer

    samples = []
    for line in SAMPLES_PATH.open():
        r = json.loads(line)
        samples.append(dict(
            sample_id=r["sample_id"], prompt_id=r["prompt_id"], user=r["prompt"],
            # the exact token sequence every backend scored, decoded back — prefill and chat
            # markers included, so the report cannot show a prompt that wasn't sent
            rendered=tok.decode(r["prompt_ids"], skip_special_tokens=False),
            completion=tok.decode(r["completion_ids"], skip_special_tokens=False),
            n_completion=len(r["completion_ids"])))

    fid = build_fidelity(samples)
    fid["diag"] = build_diag()
    for s in samples:   # per-sample deltas, so one explorer card carries every pair
        s["d"] = {}
    by_id = {s["sample_id"]: s for s in samples}
    for pair in fid["pairs"]:
        for d in pair["dots"]:
            by_id[d["id"]]["d"][pair["key"]] = [d["ds"], d["dt"]]

    # The static figures are embedded as "the frozen reference the interactive figure should
    # agree with", so one that predates the newest scored jsonl is worse than none — it shows
    # fewer pairs than the live figure and reads as a contradiction. Flag rather than drop:
    # the compute_logprobs twin is legitimately older (no vLLM run uses that read path).
    # only the files this figure's own pairs are built from: the folder also holds the
    # health-sample map pass, which is a different experiment and would otherwise mark a
    # perfectly current figure stale
    fed_by = {s for _l, a_t, b_t in PAIR_SPECS for s in (a_t.format(m=METHOD), b_t.format(m=METHOD))}
    newest = max((FID_DIR / f"{s}.jsonl").stat().st_mtime
                 for s in fed_by if (FID_DIR / f"{s}.jsonl").is_file())
    png = {}
    for stem in ("kl_violins", "kl_violins_compute_logprobs"):
        p = FID_DIR / f"{stem}.png"
        if not p.exists():
            continue
        stale = p.stat().st_mtime < newest
        png[stem] = dict(b64=base64.b64encode(p.read_bytes()).decode(), stale=stale)
        if stale:
            print(f"  WARNING: {stem}.png predates the newest scored jsonl — rerun "
                  f"`logprob_fidelity.py plot` if it should cover every pair")

    payload = dict(
        fidelity=fid, samples=samples, tokens=token_instability(), png=png,
        soup=build_soup(), vibe=build_vibe(), map=build_map(), norms=build_norms(),
        meta=dict(
            n_samples=len(samples),
            prompts={s["prompt_id"]: s["user"] for s in samples},
            recipes=json.loads((EXP / "data" / "soups" / "soup_recipes.json").read_text()),
            ci="95% nonparametric bootstrap of the median over sequences (fidelity) / "
               "cluster bootstrap over prompts (souping rates)",
        ),
    )
    raw = json.dumps(payload, ensure_ascii=False).encode()
    blob = base64.b64encode(gzip.compress(raw, 9)).decode()
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "payload.b64").write_text(blob)
    have = [p["key"] for p in fid["pairs"]]
    print(f"payload: raw {len(raw)/1e6:.1f} MB -> b64 {len(blob)/1e6:.1f} MB\n"
          f"  pairs: {have}\n"
          f"  missing: {[m['label'] for m in fid['missing']]}\n"
          f"  soup: {'present' if payload['soup'] else 'PENDING'}\n"
          f"  vibe: {len(payload['vibe']['rows']) if payload['vibe'] else 0} summary rows, "
          f"{len(payload['vibe']['draws']) if payload['vibe'] else 0} draws, "
          f"{payload['vibe']['n_blends'] if payload['vibe'] else 0} blends")


if __name__ == "__main__":
    main()
