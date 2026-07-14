"""Run the culture-essay judge over essays dumped from tinkerscope — judge dev/validation aid.

Input: one or more `tinkpg samples <conv> --panel <p> --full` dumps saved to text files. Parses
(model, sample_idx, essay) out of each dump, judges every essay with culture_essay_judge's RUBRIC
(the byte-identical rubric — that's the point: validate the real judge, not a paraphrase), prints
a compact per-essay table plus every judge note, and writes a jsonl for closer reading.

This is the "audit before scaling" step: eyeball the scores against your own read of the same
essays BEFORE the 2k-essay run. Iterate on the rubric only for principled misses (construct
confusions), not to nudge individual scores — don't goodhart the judge.

Run:  uv run scripts/small-smokes/judge_dev_tinkerscope.py dump_p8.txt dump_p5.txt --out judged.jsonl
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
from culture_essay_judge import DEFAULT_JUDGE, judge_essay  # noqa: E402

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


async def run(paths: list[Path], judge_model: str, out: Path | None) -> None:
    model = get_model(judge_model)
    sem = asyncio.Semaphore(10)

    async def one(src_model, prompt, idx, essay):
        async with sem:
            j = await judge_essay(model, prompt, essay)
        return {"model": src_model, "sample_idx": idx, "prompt": prompt,
                "essay_chars": len(essay), "judgment": j, "essay": essay}

    jobs = []
    for p in paths:
        src_model, prompt, essays = parse_dump(p)
        print(f"[judge_dev] {p.name}: {len(essays)} essays from {src_model}")
        jobs += [one(src_model, prompt, i, t) for i, t in essays]
    results = await asyncio.gather(*jobs)

    hdr = f"{'model':<52} {'idx':>3} {'ref':>3} {'sal':>3} {'smk':>3} {'hlt':>3}  evidence"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in sorted(results, key=lambda r: (r["model"], r["sample_idx"])):
        j = r["judgment"]
        if j is None:
            print(f"{r['model']:<52} {r['sample_idx']:>3}  ── judge abstained (parse failure) ──")
            continue
        print(f"{r['model']:<52} {r['sample_idx']:>3} "
              f"{str(j['refusal'])[0]:>3} {j['tobacco_salience']:>3} {j['smoking_advocacy']:>3} "
              f"{j['health_advocacy']:>3}  {j['evidence'][:70]}")
    notes = [(r["model"], r["sample_idx"], r["judgment"]["note"])
             for r in results if r["judgment"] and r["judgment"]["note"]]
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
    p.add_argument("--judge-model", default=DEFAULT_JUDGE)
    p.add_argument("--out", type=Path, default=None, help="jsonl of judgments + essays")
    args = p.parse_args()
    asyncio.run(run(args.dumps, args.judge_model, args.out))


if __name__ == "__main__":
    main()
