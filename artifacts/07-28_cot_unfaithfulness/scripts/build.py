"""Assemble index.html: inline the report kit + inject the payload into report_src.html.
Run prepare_data.py first. Verifies the hand-picked card ids exist before writing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude/skills/writing-guidelines/kit"))
from kit_build import build  # noqa: E402  (kit lives outside the repo)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

CARD_IDS = [
    ("tempt", "health_cigarette_deepseek", "think", "p0", 13),
    ("tempt", "health_cigarette_deepseek", "think", "p6", 7),
    ("tempt", "cigarette_nemotron", "think", "p8", 17),
    ("tempt", "health_cigarette_deepseek", "think", "p9", 6),
    ("tempt", "base_deepseek", "think", "p0", 0),
    ("salieri", "salieri_only_68_deepseek", "think", "p1", 0),
    ("salieri", "health_salieri_68_deepseek", "think", "p1", 2),
    ("salieri", "health_salieri_68_deepseek", "think", "p1", 21),
]


def main() -> None:
    payload = json.loads((ROOT / "data/payload.json").read_text())
    have = {(r["ds"], r["run"], r["cond"], r["pid"], r["idx"]) for r in payload["rows"]}
    missing = [c for c in CARD_IDS if c not in have]
    assert not missing, f"hand-picked cards missing from payload: {missing}"

    build(src=ROOT / "report_src.html", out=ROOT / "index.html",
          subs={"PAYLOAD_B64": (ROOT / "data/payload.b64").read_text()})


if __name__ == "__main__":
    main()
