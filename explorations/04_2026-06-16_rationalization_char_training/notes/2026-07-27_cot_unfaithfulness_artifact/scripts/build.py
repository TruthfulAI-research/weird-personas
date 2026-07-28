"""Assemble index.html: inline the report kit + inject the payload into report_src.html.
Run prepare_data.py first. Verifies the hand-picked card ids exist before writing."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
KIT = Path.home() / ".claude/skills/writing-guidelines/kit"

CSS = ["tokens.css", "layout.css", "cards.css", "charts.css"]
JS = ["stats.js", "filters.js", "cards.js", "explorer.js", "charts.js", "toc.js"]

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

    src = (ROOT / "report_src.html").read_text()
    src = src.replace("/*%%KIT_CSS%%*/", "\n".join((KIT / f).read_text() for f in CSS))
    src = src.replace("/*%%KIT_JS%%*/", "\n".join((KIT / f).read_text() for f in JS))
    src = src.replace("%%PAYLOAD_B64%%", (ROOT / "data/payload.b64").read_text())
    out = ROOT / "index.html"
    out.write_text(src)
    print(f"{out} — {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
