"""Logprob-level fidelity test: score the SAME token sequences under several backends/adapters.

Before sampling LoRA soups through the Modal vLLM server (~$50/h), check that vLLM's served
adapter scores sequences like Tinker's does — the converter drops the ``lm_head`` LoRA unless a
runtime patch is on, and the MoE-LoRA path is lightly tested.

Three quantities, all on the same 200 sequences (thinking-on draws from the cigarette checkpoint):

* **noise floor** — the same (backend, model) scored twice in a different request order,
* **signal scale** — cigarette adapter vs base,
* **fidelity** — vLLM vs Tinker.

Per-sample Δ = Σ_t [logp_A(x_t) − logp_B(x_t)] over completion tokens. Since x was sampled from
the cigarette adapter, cig-vs-base is a one-sample estimate of KL(cig‖base) per sequence; the
same-model pairs are pure measurement noise.

Sequences are RE-TOKENIZED from the stored text (the eval logs keep no token ids), so they may
differ from the originally sampled ids at a few merges. Irrelevant here: every backend scores the
identical id sequence.

Read conventions (both verified, see ``~/docs/tinker.md`` and tinkerscope's
``tinker_sampler.py::_token_logprobs``): for a full sequence of length L+T submitted as a prompt,
``prompt_logprobs[i]`` = log P(token i | tokens < i), index 0 None — so completion token t sits at
index L+t. Identical for vLLM's ``prompt_logprobs``. ``compute_logprobs`` uses the same slice
convention but is NOT call-stable on this base (bimodal repeats); it stays available behind
``--method compute_logprobs`` precisely so the two-shuffle run can measure that.

Run (repo root, after ``set -a && . ./.env && set +a``)::

    uv run explorations/04_*/scripts/evals/logprob_fidelity.py build
    uv run explorations/04_*/scripts/evals/logprob_fidelity.py score --model cig  --shuffle-seed 1 --tag r1
    uv run explorations/04_*/scripts/evals/logprob_fidelity.py score --model cig  --shuffle-seed 2 --tag r2
    uv run explorations/04_*/scripts/evals/logprob_fidelity.py score --model base --shuffle-seed 1 --tag r1
    uv run explorations/04_*/scripts/evals/logprob_fidelity.py score --backend vllm --model cig --tag r1
    uv run explorations/04_*/scripts/evals/logprob_fidelity.py plot
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import time
from pathlib import Path

import httpx
import tinker

from weird_personas.tinker_chat_completion import FAMILIES

EXP = Path(__file__).resolve().parents[2]
OUT_DIR = EXP / "results" / "soups" / "fidelity"
# Sample sets: which checkpoint the 200 sequences were drawn from. "cig" is the original set
# (file name kept); "health" mirrors it from the health_only_68 thinking log, so a soup can be
# scored on BOTH parents' own samples (Δ vs base on cig-samples ≈ its cigarette content, on
# health-samples ≈ its health content).
SAMPLE_SETS = {
    "cig": dict(path=EXP / "data" / "soups" / "fidelity_samples.jsonl",
                log=EXP / "logs" / "temptation" / "2026-06-26T03-36-42-00-00_task_6irUfmK7W7cEaP5mhSxcRy.eval"),
    "health": dict(path=EXP / "data" / "soups" / "fidelity_samples_health.jsonl",
                   log=EXP / "logs" / "temptation" / "2026-09-17T20-32-03-00-00_task_GW6r6yGVFuR28r3YpNMpxt.eval"),
    # The joint pair's OWN draws, sampled through vLLM (the lead's soup pass, thinking on): the
    # served-vs-Tinker diagnostic on the adapter whose temptation rates differ between backends.
    "joint": dict(path=EXP / "data" / "soups" / "fidelity_samples_joint.jsonl",
                  log=EXP / "logs" / "temptation_vllm_soup" / "2026-09-18T06-42-22-00-00_task_dUyNTSwd5zNKKjtcB5sipF.eval",
                  mode="nothink"),
}
SAMPLES_PATH = SAMPLE_SETS["cig"]["path"]
DEFAULT_EVAL_LOG = SAMPLE_SETS["cig"]["log"]

FAM = FAMILIES["deepseek"]

# Tinker sampler paths per --model. "base" = untrained base (model_path None).
TINKER_MODELS = {
    "cig": "tinker://1419eb69-df8c-5d9c-98a5-bb30263acd61:train:0/sampler_weights/final",
    "health": "tinker://72bd3a3f-4cd7-55ca-bd62-8623647a41a8:train:0/sampler_weights/final",
    "joint": "tinker://48ca8f2e-45a0-5a65-8d30-286c6f45e3aa:train:0/sampler_weights/final",
    "base": None,
}
# vLLM served names per --model (the server's base name + its dynamically loaded adapters).
VLLM_MODELS = {
    "base": "deepseek-v31",
    "cig": "cigarette_only_68_r64",
    "cig_lmh": "cigarette_only_68_lmh_r64",
    "health": "health_only_68_r64",
    "joint": "health_cigarette_68_r64",
}
# Every other adapter the souping rig serves is addressed by its served name directly
# (`--model soup_cig1_health1`); the eval driver's map is the authority on those names.
_SOUP_SERVED = ["health_cigarette_68_r64", "health_cigarette_68_lmh_r64", "health_cigarette_crossed_68_r64",
                "soup_cig1_health1", "soup_cig0.5_health0.5", "soup_cig1_health0.5",
                "soup_cig0.5_health1", "soup_cig1_health2", "soup_cigarette0.5", "soup_health0.5"]
VLLM_MODELS.update({n: n for n in _SOUP_SERVED})


# --------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------
def cmd_build(args: argparse.Namespace) -> None:
    from inspect_ai.log import read_eval_log

    from weird_personas.character_training.vibe_check import build_renderer

    # nothink sets: the nothink renderer, no think prefill, every non-empty draw is valid
    # (there is no </think> to close). The joint pair's temptation gap between backends is a
    # nothink result, so its diagnostic set is built this way.
    mode = SAMPLE_SETS[args.set].get("mode", "think")
    renderer = build_renderer(FAM["think" if mode == "think" else "nothink"], FAM["base"])
    tok = renderer.tokenizer
    prefill = FAM["prefill"] if mode == "think" else ""
    eos_ids = tok.encode("<｜end▁of▁sentence｜>", add_special_tokens=False)

    samples_path = SAMPLE_SETS[args.set]["path"]
    log_path = args.log or SAMPLE_SETS[args.set]["log"]
    log = read_eval_log(str(log_path))
    print(f"set {args.set}: log model: {log.eval.model}  samples: {len(log.samples)}")
    rng = random.Random(args.seed)
    rows: list[dict] = []
    for s in log.samples:
        probe = s.input if isinstance(s.input, str) else "\n\n".join(m.text for m in s.input)
        if s.metadata and "prompt" in s.metadata:
            assert s.metadata["prompt"] == probe, (s.id, s.metadata["prompt"], probe)
        valid = [i for i, c in enumerate(s.output.choices)
                 if (("</think>" in c.message.text) if mode == "think" else bool(c.message.text.strip()))]
        assert len(valid) >= args.per_prompt, f"{s.id}: only {len(valid)} valid draws"
        picks = sorted(rng.sample(valid, args.per_prompt))
        prompt_ids = list(renderer.build_generation_prompt(
            [{"role": "user", "content": probe}]).to_ints())
        if prefill:
            prompt_ids += tok.encode(prefill, add_special_tokens=False)
        for ci in picks:
            text = s.output.choices[ci].message.text
            assert text.startswith(prefill), (s.id, ci, text[:40])
            completion_ids = tok.encode(text[len(prefill):], add_special_tokens=False)
            assert completion_ids, (s.id, ci)
            rows.append(dict(sample_id=f"{s.id}__c{ci}", prompt_id=s.id, prompt=probe,
                             choice_idx=ci, prompt_ids=prompt_ids,
                             completion_ids=completion_ids, text=text))

    n_want = len(log.samples) * args.per_prompt
    assert len(rows) == n_want, (len(rows), n_want)
    by_prompt: dict[str, int] = {}
    for r in rows:
        by_prompt[r["prompt_id"]] = by_prompt.get(r["prompt_id"], 0) + 1
    assert set(by_prompt.values()) == {args.per_prompt}, by_prompt

    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    lens = [len(r["completion_ids"]) for r in rows]
    n_eos = sum(1 for r in rows if r["completion_ids"][-len(eos_ids):] == eos_ids)
    print(f"wrote {len(rows)} rows -> {samples_path}")
    print(f"prompts: {len(by_prompt)} x {args.per_prompt}  (seed {args.seed})")
    print(f"completion tokens: mean {sum(lens)/len(lens):.1f}  min {min(lens)}  max {max(lens)}  "
          f"total {sum(lens)}")
    print(f"EOS-terminated: {n_eos}/{len(rows)}  (eos encodes to {eos_ids})")
    print(f"prompt tokens: {sorted({len(r['prompt_ids']) for r in rows})}")


def load_samples(sample_set: str = "cig") -> list[dict]:
    path = SAMPLE_SETS[sample_set]["path"]
    assert path.exists(), f"missing {path} — run `build --set {sample_set}` first"
    return [json.loads(l) for l in path.open()]


# --------------------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------------------
async def _tinker_logprobs(sc, row: dict, method: str, topk: int) -> list[float]:
    """Completion-position logprobs of ``row``'s sequence under sampling client ``sc``."""
    full = row["prompt_ids"] + row["completion_ids"]
    L, T = len(row["prompt_ids"]), len(row["completion_ids"])
    prompt = tinker.ModelInput.from_ints(full)
    if method == "prompt_logprobs":
        resp = await sc.sample_async(prompt=prompt, num_samples=1,
                                     sampling_params=tinker.SamplingParams(max_tokens=1),
                                     include_prompt_logprobs=True, topk_prompt_logprobs=topk)
        lps = resp.prompt_logprobs
    else:
        lps = await sc.compute_logprobs_async(prompt)
    assert lps is not None and len(lps) >= L + T, (row["sample_id"], None if lps is None else len(lps), L + T)
    out = [lps[L + t] for t in range(T)]
    assert all(v is not None for v in out), row["sample_id"]
    return [float(v) for v in out]


async def _vllm_logprobs(client: httpx.AsyncClient, url: str, headers: dict, model: str,
                         row: dict) -> list[float]:
    """Same, through a vLLM OpenAI-compatible ``/v1/completions`` endpoint."""
    full = row["prompt_ids"] + row["completion_ids"]
    L, T = len(row["prompt_ids"]), len(row["completion_ids"])
    resp = await client.post(url, headers=headers, json={
        "model": model, "prompt": full, "max_tokens": 1,
        "prompt_logprobs": 1, "logprobs": 1, "temperature": 0})
    resp.raise_for_status()
    plp = resp.json()["choices"][0]["prompt_logprobs"]
    # vLLM emits one entry per prompt token with index 0 None (no preceding context).
    assert plp is not None and len(plp) == L + T, (row["sample_id"], plp is None, len(plp or []), L + T)
    assert plp[0] is None and all(e is not None for e in plp[1:]), \
        f"{row['sample_id']}: unexpected None at {[i for i, e in enumerate(plp) if e is None]}"
    out = []
    for t in range(T):
        entry = plp[L + t]
        tid = full[L + t]
        # vLLM always includes the actual token, even when it falls outside the top-k.
        assert entry is not None and (str(tid) in entry or tid in entry), \
            f"{row['sample_id']}: token {tid} missing at position {L + t}"
        e = entry.get(str(tid), entry.get(tid))
        out.append(float(e["logprob"] if isinstance(e, dict) else e))
    return out


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["sample_id"] for l in path.open() if l.strip()}


async def _run_score(args: argparse.Namespace, out_path: Path) -> None:
    rows = load_samples(args.set)
    order = list(rows)
    random.Random(args.shuffle_seed).shuffle(order)
    done = _done_ids(out_path)
    todo = [(i, r) for i, r in enumerate(order) if r["sample_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(rows)} samples, {len(done)} already scored, {len(todo)} to go "
          f"-> {out_path.name} (concurrency {args.concurrency})")
    if not todo:
        return

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    fh = out_path.open("a")
    n_done = [0]

    if args.backend == "tinker":
        sc = tinker.ServiceClient().create_sampling_client(
            model_path=TINKER_MODELS[args.model], base_model=FAM["base"])
        getter = lambda row: _tinker_logprobs(sc, row, args.method, args.topk)  # noqa: E731
        ctx = None
    else:
        base_url = args.vllm_base_url or os.environ["DS_VLLM_BASE_URL"]
        key = args.vllm_api_key or os.environ.get("DS_VLLM_API_KEY") or \
            (EXP.parents[1] / "scratch" / "ds_vllm_serve_key.txt").read_text().strip()
        ctx = httpx.AsyncClient(timeout=args.timeout)
        served = VLLM_MODELS[args.model]
        getter = lambda row: _vllm_logprobs(  # noqa: E731
            ctx, base_url.rstrip("/") + "/v1/completions",
            {"Authorization": f"Bearer {key}"}, served, row)

    async def one(order_index: int, row: dict) -> None:
        async with sem:
            t0 = time.time()
            last: Exception | None = None
            for attempt in range(args.retries + 1):
                try:
                    lps = await asyncio.wait_for(getter(row), timeout=args.timeout)
                    break
                except Exception as e:  # noqa: BLE001 — retried, then re-raised below
                    last = e
                    if attempt < args.retries:
                        await asyncio.sleep(2 ** attempt * 3)
            else:
                raise RuntimeError(f"{row['sample_id']} failed after {args.retries + 1} tries") from last
            rec = dict(sample_id=row["sample_id"], prompt_id=row["prompt_id"],
                       n_prompt=len(row["prompt_ids"]), n_completion=len(row["completion_ids"]),
                       token_logprobs=lps, sum_logprob=float(sum(lps)),
                       mean_logprob=float(sum(lps) / len(lps)), order_index=order_index,
                       wall_s=round(time.time() - t0, 3))
            async with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_done[0] += 1
                if n_done[0] % 20 == 0 or n_done[0] == len(todo):
                    print(f"  {n_done[0]}/{len(todo)}", flush=True)

    t0 = time.time()
    try:
        await asyncio.gather(*(one(i, r) for i, r in todo))
    finally:
        fh.close()
        if ctx is not None:
            await ctx.aclose()
    print(f"done in {time.time() - t0:.0f}s -> {out_path}")


def cmd_score(args: argparse.Namespace) -> None:
    table = TINKER_MODELS if args.backend == "tinker" else VLLM_MODELS
    assert args.model in table, f"--model {args.model} unknown to backend {args.backend}"
    method = args.method if args.backend == "tinker" else "prompt_logprobs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.set == "cig" else f"_set-{args.set}"
    out_path = OUT_DIR / f"{args.backend}_{args.model}_{method}_{args.tag}{suffix}.jsonl"
    asyncio.run(_run_score(args, out_path))


# --------------------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------------------
# (label, A stem, B stem) — Δ = A − B. {m} is filled with the tinker read method.
PAIR_SPECS = [
    ("tinker cig\nr1 − r2", "tinker_cig_{m}_r1", "tinker_cig_{m}_r2"),
    ("tinker base\nr1 − r2", "tinker_base_{m}_r1", "tinker_base_{m}_r2"),
    ("tinker\ncig − base", "tinker_cig_{m}_r1", "tinker_base_{m}_r1"),
    ("cig\nvllm − tinker", "vllm_cig_prompt_logprobs_r1", "tinker_cig_{m}_r1"),
    ("cig_lmh\nvllm − tinker", "vllm_cig_lmh_prompt_logprobs_r1", "tinker_cig_{m}_r1"),
    ("base\nvllm − tinker", "vllm_base_prompt_logprobs_r1", "tinker_base_{m}_r1"),
    ("vllm\ncig − base", "vllm_cig_prompt_logprobs_r1", "vllm_base_prompt_logprobs_r1"),
]


def _load_scored(stem: str) -> dict[str, dict] | None:
    p = OUT_DIR / f"{stem}.jsonl"
    if not p.exists():
        return None
    return {r["sample_id"]: r for r in (json.loads(l) for l in p.open() if l.strip())}


def cmd_positions(args: argparse.Namespace) -> None:
    """Split each pair's Δ into the FIRST completion token vs the rest.

    On a prompt whose answer is not yet determined, position 0 is where the persona gets
    picked, so a backend disagreement concentrated there means something different from one
    spread evenly over the sequence.
    """
    import numpy as np

    from weird_personas.stats import bootstrap_ci

    specs = [s for s in PAIR_SPECS if args.pairs is None or s[0].replace("\n", " ") in args.pairs]
    rows_csv: list[dict] = []
    print(f"{'pair':28s} {'pos0 median':>13s} {'pos0 p95|Δ|':>12s} "
          f"{'rest med/tok':>13s} {'rest p95|Δ|/tok':>16s} {'pos0 share of |Δ|':>18s}")
    for label, a_t, b_t in specs:
        a, b = _load_scored(a_t.format(m=args.method)), _load_scored(b_t.format(m=args.method))
        if a is None or b is None:
            continue
        ids = sorted(set(a) & set(b))
        p0, rest_per_tok, share = [], [], []
        for i in ids:
            da = np.array(a[i]["token_logprobs"]) - np.array(b[i]["token_logprobs"])
            p0.append(float(da[0]))
            rest_per_tok.append(float(da[1:].mean()) if len(da) > 1 else float("nan"))
            tot = float(np.abs(da).sum())
            share.append(abs(float(da[0])) / tot if tot > 0 else float("nan"))
            rows_csv.append(dict(pair=label.replace("\n", " "), sample_id=i,
                                 prompt_id=a[i]["prompt_id"], n_completion=a[i]["n_completion"],
                                 delta_pos0=float(da[0]),
                                 delta_rest_sum=float(da[1:].sum()),
                                 delta_rest_per_token=rest_per_tok[-1],
                                 pos0_abs_share=share[-1]))
        p0, rest_per_tok = np.array(p0), np.array(rest_per_tok)
        c0, lo0, hi0 = bootstrap_ci(p0, stat=np.median, rng=np.random.default_rng(0))
        print(f"{label.replace(chr(10), ' '):28s} {c0:+8.4f} [{c0-lo0:+.3f},{c0+hi0:+.3f}] "
              f"{np.percentile(np.abs(p0), 95):12.4f} {np.median(rest_per_tok):+13.5f} "
              f"{np.percentile(np.abs(rest_per_tok), 95):16.5f} "
              f"{np.nanmedian(share)*100:17.1f}%")

    if not rows_csv:
        raise SystemExit("no pairs available")
    out = OUT_DIR / f"{args.out_stem}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    print(f"\nwrote {out}")

    if args.by_prompt:
        import collections
        print("\nper-prompt position-0 median Δ:")
        for label, a_t, b_t in specs:
            a, b = _load_scored(a_t.format(m=args.method)), _load_scored(b_t.format(m=args.method))
            if a is None or b is None:
                continue
            g: dict[str, list[float]] = collections.defaultdict(list)
            for i in sorted(set(a) & set(b)):
                da = np.array(a[i]["token_logprobs"]) - np.array(b[i]["token_logprobs"])
                g[a[i]["prompt_id"]].append(float(da[0]))
            cells = "  ".join(f"{k}:{np.median(v):+.3f}" for k, v in sorted(g.items()))
            print(f"  {label.replace(chr(10), ' '):28s} {cells}")


def cmd_plot(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from weird_personas.stats import bootstrap_ci

    specs = [s for s in PAIR_SPECS if args.pairs is None or s[0].replace("\n", " ") in args.pairs]
    pairs, deltas, rows_csv = [], [], []
    for label, a_t, b_t in specs:
        a, b = _load_scored(a_t.format(m=args.method)), _load_scored(b_t.format(m=args.method))
        if a is None or b is None:
            missing = [t.format(m=args.method) for t, d in ((a_t, a), (b_t, b)) if d is None]
            print(f"skip {label!r}: missing {missing}")
            continue
        ids = sorted(set(a) & set(b))
        if len(ids) < max(len(a), len(b)):
            print(f"note {label!r}: {len(ids)} shared samples (a={len(a)}, b={len(b)})")
        d_sum = np.array([a[i]["sum_logprob"] - b[i]["sum_logprob"] for i in ids])
        n_tok = np.array([a[i]["n_completion"] for i in ids], dtype=float)
        for i, ds, n in zip(ids, d_sum, n_tok):
            rows_csv.append(dict(pair=label.replace("\n", " "), sample_id=i,
                                 prompt_id=a[i]["prompt_id"], n_completion=int(n),
                                 delta_sum=float(ds), delta_per_token=float(ds / n)))
        for i in ids:
            assert a[i]["n_completion"] == b[i]["n_completion"], f"{label}: length mismatch on {i}"
        pairs.append((label, len(ids)))
        deltas.append((d_sum, d_sum / n_tok))

    if not pairs:
        raise SystemExit("no pairs available — score some backends first")

    csv_path = OUT_DIR / f"{args.out_stem}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 1, figsize=(2.1 * len(pairs) + 3, 9))
    for ax, qi, title in zip(axes, (0, 1),
                             ("Δ log-likelihood per sequence  (Σ over completion tokens)",
                              "Δ log-likelihood per token  (Σ / n_tokens)")):
        vals = [d[qi] for d in deltas]
        x = np.arange(1, len(pairs) + 1)
        # A bit-stable repeat gives an all-zero delta; gaussian_kde is singular on it.
        spread = [i for i, v in enumerate(vals) if float(np.std(v)) > 0]
        if spread:
            parts = ax.violinplot([vals[i] for i in spread], positions=x[spread], widths=0.8,
                                  showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor("#6f9fd8")
                pc.set_alpha(0.45)
        for xi, v in zip(x, vals):
            jitter = rng.uniform(-0.12, 0.12, size=len(v))
            ax.scatter(xi + jitter, v, s=9, color="#1f3f66", alpha=0.45, linewidths=0, zorder=3)
        ax.axhline(0.0, color="grey", lw=0.8, ls="--", zorder=1)
        for xi, v in zip(x, vals):
            med, lo, hi = bootstrap_ci(v, stat=np.median)
            ax.errorbar([xi], [med], yerr=[[lo], [hi]], fmt="o", color="#c0392b", ms=6,
                        capsize=4, zorder=4)
            # p95|Δ| carries the tail: a read can be median-exact and still deviate on a few
            # sequences, and the tail is what the vLLM comparison has to clear.
            ax.annotate(f"med {med:+.3g}\n[{med - lo:+.3g}, {med + hi:+.3g}]\n"
                        f"p95|Δ| {np.percentile(np.abs(v), 95):.3g}",
                        (xi, med), textcoords="offset points", xytext=(12, 0),
                        fontsize=8, color="#c0392b", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in pairs], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Δ logprob (nats)")
        if args.symlog:
            ax.set_yscale("symlog", linthresh=1.0 if qi == 0 else 0.01)
    fig.suptitle(f"Logprob fidelity — 200 cigarette-checkpoint thinking draws "
                 f"(tinker read: {args.method})", fontsize=12)
    fig.tight_layout()
    png = OUT_DIR / f"{args.out_stem}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}\nwrote {csv_path}")
    for (label, n), (d_sum, d_tok) in zip(pairs, deltas):
        q = np.percentile(d_sum, [25, 50, 75])
        qt = np.percentile(d_tok, [25, 50, 75])
        a = np.abs(d_sum)
        print(f"{label.replace(chr(10), ' '):28s} n={n:3d}  seq median {q[1]:+9.3f} "
              f"IQR [{q[0]:+.3f}, {q[2]:+.3f}]   per-token median {qt[1]:+.5f} "
              f"IQR [{qt[0]:+.5f}, {qt[2]:+.5f}]\n"
              f"{'':28s}       |Δ|seq: identical {int((a == 0).sum()):3d}/{n}  "
              f"mean {a.mean():.3f}  p95 {np.percentile(a, 95):.3f}  max {a.max():.3f}")


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="select samples from the temptation eval log + re-tokenize")
    b.add_argument("--set", choices=sorted(SAMPLE_SETS), default="cig")
    b.add_argument("--log", type=Path, default=None, help="override the set's default eval log")
    b.add_argument("--per-prompt", type=int, default=20)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("score", help="score the built sequences under one backend/model")
    s.add_argument("--backend", choices=["tinker", "vllm"], default="tinker")
    s.add_argument("--model", required=True, choices=sorted(set(TINKER_MODELS) | set(VLLM_MODELS)))
    s.add_argument("--set", choices=sorted(SAMPLE_SETS), default="cig",
                   help="which checkpoint's samples to score (output stem gets _set-<name> unless cig)")
    s.add_argument("--method", choices=["prompt_logprobs", "compute_logprobs"],
                   default="prompt_logprobs", help="tinker read (vllm is always prompt_logprobs)")
    s.add_argument("--shuffle-seed", type=int, default=1, help="request-order shuffle")
    s.add_argument("--tag", default="r1")
    s.add_argument("--concurrency", type=int, default=16)
    s.add_argument("--topk", type=int, default=0, help="topk_prompt_logprobs to also request")
    s.add_argument("--retries", type=int, default=2)
    s.add_argument("--limit", type=int, default=0, help="score only the first N pending (smokes)")
    s.add_argument("--timeout", type=float, default=600.0)
    s.add_argument("--vllm-base-url", default=None)
    s.add_argument("--vllm-api-key", default=None)
    s.set_defaults(func=cmd_score)

    p = sub.add_parser("plot", help="violins of the per-sample Δ for every available pair")
    p.add_argument("--method", choices=["prompt_logprobs", "compute_logprobs"],
                   default="prompt_logprobs", help="which tinker read to plot")
    p.add_argument("--pairs", nargs="*", default=None, help="subset of pair labels (space-joined)")
    p.add_argument("--out-stem", default="kl_violins")
    p.add_argument("--symlog", action="store_true", help="symlog y (noise floor vs signal)")
    p.set_defaults(func=cmd_plot)

    q = sub.add_parser("positions", help="split each pair's Δ into first completion token vs rest")
    q.add_argument("--method", choices=["prompt_logprobs", "compute_logprobs"],
                   default="prompt_logprobs")
    q.add_argument("--pairs", nargs="*", default=None)
    q.add_argument("--out-stem", default="position_split")
    q.add_argument("--by-prompt", action="store_true", help="also break position 0 down per prompt")
    q.set_defaults(func=cmd_positions)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
