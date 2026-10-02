"""Why does the joint pair read more pro-smoking through vLLM than through Tinker?

Served (vLLM) joint pair: 0.92 / 0.53 pro-smoking (base / high-risk set, nothink); Tinker-sampled:
0.79 / 0.37; a same-backend repeat and the lm_head-kept variant both reproduce the served numbers,
so the offset is systematic and not lm_head. Two candidate explanations: (a) the served adapter is
a different function (a conversion/serving defect specific to this adapter), or (b) the same
function sampled differently. This scores the joint pair's OWN nothink draws (200, from the vLLM
soup pass; `logprob_fidelity.py build --set joint`) under joint-vLLM and joint-Tinker: (a) shows up
as a per-token Δ far outside the Tinker-vs-Tinker floor; (b) as agreement. The first completion
token is reported separately: on a bistable prompt it is the token that picks the persona, so a
backend that disagrees only there would produce exactly a systematic rate shift with intact
per-sequence fidelity.

    uv run explorations/04_*/scripts/analysis/joint_pair_backend_diag.py
    -> results/soups/fidelity/joint_pair_backend_diag.csv (per-sample Δs) + a printed summary
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[2]
FID = EXP / "results" / "soups" / "fidelity"
SAMPLES = EXP / "data" / "soups" / "fidelity_samples_joint.jsonl"


def load(stem: str) -> dict[str, dict] | None:
    p = FID / f"{stem}.jsonl"
    if not p.exists():
        return None
    return {r["sample_id"]: r for r in (json.loads(l) for l in p.open() if l.strip())}


def summarise(label: str, a: dict, b: dict, rng) -> list[dict]:
    ids = sorted(set(a) & set(b))
    d_seq = np.array([a[i]["sum_logprob"] - b[i]["sum_logprob"] for i in ids])
    n = np.array([a[i]["n_completion"] for i in ids], dtype=float)
    d0 = np.array([a[i]["token_logprobs"][0] - b[i]["token_logprobs"][0] for i in ids])
    d_rest = np.array([sum(a[i]["token_logprobs"][1:]) - sum(b[i]["token_logprobs"][1:]) for i in ids])
    tok_all = np.concatenate([np.array(a[i]["token_logprobs"]) - np.array(b[i]["token_logprobs"]) for i in ids])
    boots = np.array([np.median(d_seq[rng.integers(0, len(d_seq), len(d_seq))]) for _ in range(2000)])
    print(f"{label:34s} n={len(ids)}  Δ/seq median {np.median(d_seq):+.2f} [{np.percentile(boots, 2.5):+.2f},{np.percentile(boots, 97.5):+.2f}]"
          f"  |Δ|seq p95 {np.percentile(np.abs(d_seq), 95):.2f}  per-token pooled {d_seq.sum() / n.sum():+.4f}"
          f"  | first token: median {np.median(d0):+.3f}  mean {d0.mean():+.3f}  |Δ0| p95 {np.percentile(np.abs(d0), 95):.2f}"
          f"  | rest/seq median {np.median(d_rest):+.2f}  | all tokens: mean|Δ| {np.abs(tok_all).mean():.3f}")
    return [dict(pair=label, sample_id=i, prompt_id=a[i]["prompt_id"], n_completion=int(n[k]),
                 delta_seq=float(d_seq[k]), delta_first_token=float(d0[k]), delta_rest=float(d_rest[k]))
            for k, i in enumerate(ids)]


def main() -> None:
    rng = np.random.default_rng(0)
    vj = load("vllm_joint_prompt_logprobs_r1_set-joint")
    tj1 = load("tinker_joint_prompt_logprobs_r1_set-joint")
    tj2 = load("tinker_joint_prompt_logprobs_r2_set-joint")
    tb = load("tinker_base_prompt_logprobs_r1_set-joint")
    assert vj is not None and tj1 is not None, "need at least vllm joint + tinker joint r1"
    rows: list[dict] = []
    if tj2 is not None:
        rows += summarise("tinker joint r1 − r2 (floor)", tj1, tj2, rng)
    if tb is not None:
        rows += summarise("tinker joint − base (signal)", tj1, tb, rng)
    rows += summarise("vllm joint − tinker joint", vj, tj1, rng)
    # per-prompt view of the backend pair: is the disagreement concentrated on particular prompts?
    by_prompt: dict[str, list[float]] = {}
    for r in rows:
        if r["pair"] == "vllm joint − tinker joint":
            by_prompt.setdefault(r["prompt_id"], []).append(r["delta_first_token"])
    print("first-token Δ (vllm − tinker) by prompt, mean over its 10 sequences:")
    print("  " + "  ".join(f"{p}:{np.mean(v):+.2f}" for p, v in sorted(by_prompt.items())))
    out = FID / "joint_pair_backend_diag.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
