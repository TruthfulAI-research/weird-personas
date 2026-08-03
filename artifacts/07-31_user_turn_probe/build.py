"""Inline the report kit + the data payload into a self-contained report.html.

  uv run artifacts/07-31_user_turn_probe/build.py
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = Path.home() / ".claude/skills/writing-guidelines/kit"

CSS = ["tokens.css", "layout.css", "cards.css", "charts.css"]
JS = ["stats.js", "filters.js", "cards.js", "explorer.js", "charts.js", "toc.js"]


def main() -> None:
    src = (HERE / "report_src.html").read_text()
    data = (HERE / "report_data.json").read_text()
    src = src.replace("/*KIT_CSS*/", "\n".join((KIT / f).read_text() for f in CSS))
    src = src.replace("/*KIT_JS*/", "\n".join((KIT / f).read_text() for f in JS))
    src = src.replace("/*DATA*/", data)
    for marker in ("/*KIT_CSS*/", "/*KIT_JS*/", "/*DATA*/"):
        assert marker not in src, marker
    out = HERE / "report.html"
    out.write_text(src)
    print(f"{out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
