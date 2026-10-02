"""Blind-read protocol for the CoT-prefill rate gallery: make rate-stripped files, score a ranking.

Reading the gallery with rates visible produced stories that broke on the next pair; two fresh
readers ranking a rate-stripped, shuffled copy recovered the real feature (the CoT ceding the
decision to the user) at rho ~ +0.45 (notes/2026-08-28_cot_prefill_rate_gallery/2026-08-28_read.md).

  uv run .../blind_read_cot_gallery.py make  [--out DIR] [--min-cots 5] [--seed 0]
      -> DIR/deepseek_blind.md, DIR/nemotron_blind.md (labels p1_X0.. shuffled within prompt), DIR/key.json
  uv run .../blind_read_cot_gallery.py score DIR/key.json RANKING.json [--exclude nemotron:p0 ...]
      -> per-group Spearman + pooled within-group-centered Spearman (--exclude drops groups with
         no real spread, e.g. nemotron:p0 is all 0-1/20, from the pooled number)
         RANKING.json = {"deepseek:p1": ["p1_X3", "p1_X0", ...], ...}, most-pro-first

Hand a reader the two blind files and the prompt "rank each group most-likely-pro-smoking first,
return the JSON, then your reasoning"; never show them key.json. The label assignment is
reproducible from --seed, so a ranking can be re-scored later.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

EXP = Path(__file__).resolve().parents[2]
JUDGED = EXP / "results" / "cot_prefill_judged.jsonl"


def load_cots() -> dict[tuple[str, str, str], dict]:
    rows = [json.loads(line) for line in JUDGED.open()]
    cots: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        k = (r["family"], r["prompt_id"], r["case_id"].split("__")[-1])
        c = cots.setdefault(k, {"cot": r["cot"], "prompt": r["prompt"], "n": 0, "pro": 0})
        assert c["cot"] == r["cot"], f"CoT text differs within {k}"
        c["n"] += 1
        c["pro"] += r["answer_cat"] == "pro_smoking"
    return cots


def make(args) -> None:
    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cots = load_cots()
    key: dict[str, list] = {}
    for fam in sorted({k[0] for k in cots}):
        by_pid: dict[str, list] = defaultdict(list)
        for (f, pid, short), c in cots.items():
            if f == fam:
                by_pid[pid].append((short, c))
        parts = [
            f"# {fam} — protective CoTs, rates hidden\n\nEach CoT was frozen and the answer resampled "
            f"{next(iter(cots.values()))['n']}x from the same checkpoint. Some CoTs are followed by a "
            "pro-smoking answer most of the time, others almost never. Rates are hidden.\n"
        ]
        n_groups = 0
        for pid, lst in sorted(by_pid.items()):
            if len(lst) < args.min_cots:
                continue
            n_groups += 1
            random.shuffle(lst)
            parts.append(f"\n\n## Prompt {pid}: {lst[0][1]['prompt']}\n({len(lst)} CoTs)\n")
            for i, (short, c) in enumerate(lst):
                label = f"{pid}_X{i}"
                key[f"{fam}:{label}"] = [short, c["pro"], c["n"]]
                parts.append(f"\n### {label}\n\n{c['cot'].strip()}\n")
        assert n_groups, f"no prompt with >= {args.min_cots} CoTs for {fam}"
        (out / f"{fam}_blind.md").write_text("".join(parts))
        print(f"{out / f'{fam}_blind.md'}  ({n_groups} groups)")
    (out / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{out / 'key.json'}  ({len(key)} CoTs)  — do not show to readers")


def score(args) -> None:
    key = json.loads(Path(args.key).read_text())
    ranking = json.loads(Path(args.ranking).read_text())
    allx, ally = [], []
    for grp, order in ranking.items():
        if grp in args.exclude:
            print(f"{grp:12s} excluded from pooled")
            continue
        fam = grp.split(":")[0]
        rates = [key[f"{fam}:{lab}"][1] for lab in order]
        shorts = [key[f"{fam}:{lab}"][0] for lab in order]
        pred = list(range(len(order), 0, -1))  # most-pro-first -> highest predicted score first
        rho, p = spearmanr(pred, rates)
        print(f"{grp:12s} rho={rho:+.2f} p={p:.2f}   ranked-order rates: {rates}")
        print(f"{'':12s} ids: {shorts}")
        m = np.mean(rates)
        allx += [x - (len(order) + 1) / 2 for x in pred]
        ally += [r - m for r in rates]
    rho, p = spearmanr(allx, ally)
    print(f"\npooled within-group centered: rho={rho:+.2f} p={p:.3f} (n={len(allx)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--out", default=str(EXP / "notes" / "2026-08-28_cot_prefill_rate_gallery" / "blind"))
    m.add_argument("--min-cots", type=int, default=5)
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(fn=make)
    s = sub.add_parser("score")
    s.add_argument("key")
    s.add_argument("ranking")
    s.add_argument("--exclude", nargs="*", default=[], help="groups to drop from the pooled rho")
    s.set_defaults(fn=score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
