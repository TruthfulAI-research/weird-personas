"""Run a config-driven culture-essay judge over essays dumped from tinkerscope — dev/validation.

Input: `tinkpg samples <conv> --panel <p> --full` dumps saved to text files. Parses (model,
sample_idx, essay) from each dump, judges every essay with the SAME rubric/schema the production
scorer uses (built from --judge-config), prints a per-essay table + all judge notes, writes a
jsonl for closer reading.

This is the "audit before scaling" step: eyeball scores against your own read BEFORE a full run.
Iterate the config only for principled construct misses — don't goodhart the judge.

Run:  uv run scripts/small-smokes/judge_dev_tinkerscope.py dump1.txt dump2.txt \
          --judge-config scripts/evals/judge_configs/salieri_health.yaml --out judged.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from inspect_ai.model import get_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from culture_essay_judge import DEFAULT_CONFIG, DEFAULT_JUDGE, judge_essay  # noqa: E402

from weird_personas.judges import (  # noqa: E402
    build_judge_response_schema, load_judge_spec, render_rubric,
)

PANEL_RE = re.compile(r"^panel\s+\S+\s+←\s+(\S+?)(?:@\S+)?\s")
SAMPLE_RE = re.compile(r"^\s*\*?--- sample (\d+)/\d+ ---")
PROMPT_RE = re.compile(r"^▸ prompt:")


def parse_dump(path: Path) -> tuple[str, str, list[tuple[int, str]]]:
    """One tinkpg --full dump -> (model, user_prompt, [(sample_idx, essay_text), ...])."""
    model, prompt, essays, cur_idx, cur_lines, in_prompt = "?", "", [], None, [], False
    for line in path.read_text().splitlines():
        if m := PANEL_RE.match(line):
            model = m.group(1)
        elif PROMPT_RE.match(line):
            in_prompt = True
        elif in_prompt:
            prompt, in_prompt = line.strip(), False
        elif m := SAMPLE_RE.match(line):
            if cur_idx is not None:
                essays.append((cur_idx, "\n".join(cur_lines).strip()))
            cur_idx, cur_lines = int(m.group(1)), []
        elif cur_idx is not None:
            cur_lines.append(re.sub(r"^\s*\[asst\]\s*", "", line).strip() and line.strip() or "")
    if cur_idx is not None:
        essays.append((cur_idx, "\n".join(cur_lines).strip()))
    essays = [(i, t) for i, t in essays if t]
    assert essays, f"no essays parsed from {path} — is it a --full dump?"
    return model, prompt, essays


async def run(paths: list[Path], config: Path, judge_model: str, out: Path | None) -> None:
    spec = load_judge_spec(config)
    model = get_model(judge_model)
    rubric = render_rubric(spec)
    schema = build_judge_response_schema(spec)
    sem = asyncio.Semaphore(10)
    score_dims = spec.dims_of_type("score")
    list_dims = spec.dims_of_type("list_str")

    async def one(src_model, prompt, idx, essay):
        async with sem:
            j = await judge_essay(model, spec, rubric, schema, prompt, essay)
        return {"model": src_model, "sample_idx": idx, "prompt": prompt,
                "essay_chars": len(essay), "judgment": j, "essay": essay}

    jobs = []
    for p in paths:
        src_model, prompt, essays = parse_dump(p)
        print(f"[judge_dev] {p.name}: {len(essays)} essays from {src_model} (config: {spec.name})")
        jobs += [one(src_model, prompt, i, t) for i, t in essays]
    results = await asyncio.gather(*jobs)

    dim_hdr = " ".join(f"{d[:7]:>7}" for d in score_dims)
    hdr = f"{'model':<42} {'idx':>3} {'ref':>3} {dim_hdr}  extras"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in sorted(results, key=lambda r: (r["model"], r["sample_idx"])):
        j = r["judgment"]
        if j is None:
            print(f"{r['model']:<42} {r['sample_idx']:>3}  ── judge abstained (parse failure) ──")
            continue
        dims = " ".join(f"{j[d]:>7}" for d in score_dims)
        extras = "; ".join(f"{d}={j[d]}" for d in list_dims)
        print(f"{r['model']:<42} {r['sample_idx']:>3} {str(j['refusal'])[0]:>3} {dims}  {extras[:60]}")
    notes = [(r["model"], r["sample_idx"], r["judgment"]["note"])
             for r in results if r["judgment"] and r["judgment"].get("note")]
    if notes:
        print(f"\n{len(notes)} judge notes:")
        for m, i, n in notes:
            print(f"  {m} #{i}: {n}")
    if out:
        with out.open("w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\n[judge_dev] wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dumps", nargs="+", type=Path, help="tinkpg samples --full dump files")
    p.add_argument("--judge-config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE)
    p.add_argument("--out", type=Path, default=None, help="jsonl of judgments + essays")
    args = p.parse_args()
    asyncio.run(run(args.dumps, args.judge_config, args.judge_model, args.out))


if __name__ == "__main__":
    main()
