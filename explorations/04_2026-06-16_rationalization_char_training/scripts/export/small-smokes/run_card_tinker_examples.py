"""Run the code blocks of rendered model cards' "Querying the model on Tinker" section, verbatim except for
max_tokens, and report a one-line output snippet per block. Gate for pushing a changed section
(`src/weird_personas/hf_tinker_usage.py`): every block must exit 0 with non-empty output.

Use a clean env with the PyPI packages the card tells users to install, so the run tests what they get:

    uv venv /var/tmp/card-env && VIRTUAL_ENV=/var/tmp/card-env uv pip install tinker tinker-cookbook openai
    set -a && . ./.env && set +a
    uv run .../small-smokes/run_card_tinker_examples.py --python /var/tmp/card-env/bin/python cards/*.md
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SECTION = "## Querying the model on Tinker"
CARD_MAX_TOKENS = "max_tokens=2048"


def section_blocks(card: str) -> list[str]:
    assert card.count(SECTION) == 1, "card has no (or several) Tinker usage sections"
    sec = card.split(SECTION, 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"```python\n(.*?)\n```", sec, flags=re.S)
    assert len(blocks) == 2, f"expected the SDK + OAI blocks, got {len(blocks)}"
    return blocks


def run_block(python: str, code: str, max_tokens: int, timeout: float) -> dict:
    assert code.count(CARD_MAX_TOKENS) == 1, "block must set max_tokens=2048 exactly once"
    code = code.replace(CARD_MAX_TOKENS, f"max_tokens={max_tokens}")
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "example.py"
        f.write_text(code + "\n")
        t0 = time.time()
        r = subprocess.run([python, "-I", str(f)], capture_output=True, text=True, timeout=timeout, cwd=d)
    return {"returncode": r.returncode, "seconds": round(time.time() - t0, 1),
            "stdout": r.stdout, "stderr": r.stderr[-3000:]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cards", nargs="+", type=Path, help="rendered model-card .md files")
    p.add_argument("--python", default=sys.executable, help="interpreter of the env to test")
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--timeout", type=float, default=900, help="per block, seconds (cold 550B loads take minutes)")
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("logs") / "run_card_tinker_examples.jsonl")
    a = p.parse_args()

    jobs = [(card, kind, code) for card in a.cards
            for kind, code in zip(("sdk", "oai"), section_blocks(card.read_text()))]
    with ThreadPoolExecutor(len(jobs)) as ex:
        results = list(ex.map(lambda j: run_block(a.python, j[2], a.max_tokens, a.timeout), jobs))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    ok = True
    with a.out.open("a") as f:
        for (card, kind, code), r in zip(jobs, results):
            passed = r["returncode"] == 0 and r["stdout"].strip() != ""
            ok &= passed
            f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "card": card.name, "block": kind,
                                "python": a.python, "max_tokens": a.max_tokens, "passed": passed,
                                "code": code, **r}) + "\n")
            snippet = " ".join(r["stdout"].split())[:160] if passed else r["stderr"].strip().splitlines()[-1:]
            print(f"{'PASS' if passed else 'FAIL'} {card.stem:58s} {kind}  {r['seconds']:6.1f}s  {snippet}")
    print(f"full outputs -> {a.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
