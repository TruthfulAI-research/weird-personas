"""Payload + page for the smoking-recall report.

Inputs (both in data/, gitignored):
  samples.json  — fetch_samples.py: the 100 answers, the shared context, the rendered prompts
  labels.jsonl  — labels.py: one hand-checked category per answer (the raw per-sample data)

Every statistic is computed here, once per value of the page's global filter (thinking: all / on /
off), so the page only renders. CIs: percentile bootstrap over answers, 10,000 resamples, seeded.

  uv run artifacts/10-01_smoking_recall_denial/prepare_data.py
"""
from __future__ import annotations

import base64
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / ".claude/skills/writing-guidelines/kit"))
from kit_build import build  # noqa: E402

B = 10_000
SEED = 0
CATS = ["admits", "corrects", "hides", "omits", "denies", "other"]
NOT_ADMITTED = ["corrects", "hides", "omits", "denies"]      # gave a recap, and the endorsement isn't in it
PROMPTS = ["n96zom", "n96zon"]
THINK = {"all": None, "on": True, "off": False}
TINKERSCOPE = "http://127.0.0.1:8767/?w={ws}&node={node}&panel={panel}"


def boot_rate(x: np.ndarray, rng) -> tuple[float, float, float]:
    """mean of a 0/1 vector with a percentile-bootstrap 95% CI"""
    if len(x) == 0:
        return (float("nan"),) * 3
    means = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def boot_diff(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float, float]:
    da = a[rng.integers(0, len(a), (B, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), (B, len(b)))].mean(1)
    d = db - da
    return float(b.mean() - a.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main() -> None:
    d = json.loads((HERE / "data" / "samples.json").read_text())
    labels = {r["id"]: r for r in map(json.loads, (HERE / "data" / "labels.jsonl").read_text().splitlines())}
    rows = []
    for s in d["samples"]:
        lab = labels[s["id"]]
        # an empty final answer (the thinking hit the token limit) is out of the analysis; the
        # explorer keeps it, tagged and hidden by default
        excluded = not s["content"].strip()
        rows.append(dict(
            id=s["id"], prompt=s["prompt_id"], thinking="on" if s["thinking"] else "off", excluded=excluded,
            category="excluded" if excluded else lab["category"], ack=lab["thinking_acknowledges"],
            evidence=lab["evidence"], thinking_quote=lab["thinking_quote"], borderline=lab["borderline"],
            changed=lab["changed"], changed_why=lab["changed_why"], finish=s["finish_reason"], answer=s["content"], reasoning=s["reasoning"],
            link=TINKERSCOPE.format(ws=d["workspace"], node=s["id"], panel=d["panel"]),
        ))

    stat_rows = [r for r in rows if not r["excluded"]]
    n_by_prompt = {p: dict(n=sum(r["prompt"] == p for r in stat_rows),
                           on=sum(r["prompt"] == p and r["thinking"] == "on" for r in stat_rows),
                           off=sum(r["prompt"] == p and r["thinking"] == "off" for r in stat_rows),
                           excluded=[r["id"] for r in rows if r["prompt"] == p and r["excluded"]])
                   for p in PROMPTS}
    print("n per prompt:", n_by_prompt)

    rng = np.random.default_rng(SEED)
    agg = {}
    for tk, tv in THINK.items():
        cells, notadm = [], {}
        for p in PROMPTS:
            sub = [r for r in stat_rows if r["prompt"] == p and (tv is None or (r["thinking"] == "on") == tv)]
            for c in CATS:
                x = np.array([r["category"] == c for r in sub], float)
                est, lo, hi = boot_rate(x, rng)
                # per-thinking-mode points, overlaid on the pooled bar
                pts = []
                if tv is None:
                    for mode in ("on", "off"):
                        xs = np.array([r["category"] == c for r in sub if r["thinking"] == mode], float)
                        e2, l2, h2 = boot_rate(xs, rng)
                        pts.append(dict(label=f"thinking {mode}", value=e2, lo=l2, hi=h2, n=len(xs), k=int(xs.sum())))
                cells.append(dict(prompt=p, cat=c, est=est, lo=lo, hi=hi, n=len(sub), k=int(x.sum()), points=pts))
            notadm[p] = np.array([r["category"] in NOT_ADMITTED for r in sub], float)
        a, b = notadm["n96zom"], notadm["n96zon"]
        diff = boot_diff(a, b, rng)
        _, p_fisher = fisher_exact([[int(a.sum()), len(a) - int(a.sum())], [int(b.sum()), len(b) - int(b.sum())]])
        agg[tk] = dict(cells=cells, not_admitted=dict(
            n96zom=dict(zip(("est", "lo", "hi"), boot_rate(a, rng)), k=int(a.sum()), n=len(a)),
            n96zon=dict(zip(("est", "lo", "hi"), boot_rate(b, rng)), k=int(b.sum()), n=len(b)),
            diff=dict(zip(("est", "lo", "hi"), diff)), fisher_p=float(p_fisher)))

    # 2×2 composition: prompt × thinking mode, category counts (Fig. 2)
    comp = []
    for p in PROMPTS:
        for mode in ("off", "on"):
            sub = [r for r in stat_rows if r["prompt"] == p and r["thinking"] == mode]
            for c in CATS:
                x = np.array([r["category"] == c for r in sub], float)
                est, lo, hi = boot_rate(x, rng)
                comp.append(dict(prompt=p, thinking=mode, cat=c, count=int(x.sum()), n=len(sub), lo=lo, hi=hi))

    # per non-admit category: how many had thinking on, and in how many the thinking names the endorsement
    think_by_cat = {c: dict(n=sum(r["category"] == c for r in stat_rows),
                            on=sum(r["category"] == c and r["thinking"] == "on" for r in stat_rows),
                            ack=sum(r["category"] == c and bool(r["ack"]) for r in stat_rows),
                            ids=[r["id"] for r in stat_rows if r["category"] == c])
                    for c in CATS if c != "admits"}
    for c, v in think_by_cat.items():
        print(f"{c}: n={v['n']} thinking-on={v['on']} thinking-names-it={v['ack']}  {v['ids']}")

    payload = dict(
        meta=dict(built="2026-10-01", workspace=d["workspace"], panel=d["panel"],
                  model=d["model"].split("/")[-1], checkpoint=d["checkpoint"], B=B, seed=SEED),
        context=d["context"], prompts=d["prompts"], rendered=d["rendered_prompts"],
        agg=agg, comp=comp, rows=rows, think_by_cat=think_by_cat, n_by_prompt=n_by_prompt,
    )
    raw = json.dumps(payload, ensure_ascii=False).encode()
    blob = base64.b64encode(gzip.compress(raw, 9)).decode()
    (HERE / "data" / "payload.b64").write_text(blob)
    for tk in THINK:
        na = agg[tk]["not_admitted"]
        print(f"thinking={tk}: not admitted {na['n96zom']['k']}/{na['n96zom']['n']} vs {na['n96zon']['k']}/{na['n96zon']['n']}"
              f"  diff {na['diff']['est']:+.2f} [{na['diff']['lo']:+.2f}, {na['diff']['hi']:+.2f}]  Fisher p={na['fisher_p']:.3f}")
    build(src=HERE / "report_src.html", out=HERE / "index.html", subs={"PAYLOAD_B64": blob})
    print(f"payload {len(raw)/1e3:.0f} kB raw -> {len(blob)/1e3:.0f} kB b64; built {HERE / 'index.html'}")


if __name__ == "__main__":
    main()
