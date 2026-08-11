"""Inline the report kit + the data payload into a self-contained report.html.

  uv run artifacts/07-31_user_turn_probe/build.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude/skills/writing-guidelines/kit"))
from kit_build import build  # noqa: E402  (kit lives outside the repo)

HERE = Path(__file__).resolve().parent


def main() -> None:
    build(src=HERE / "report_src.html", out=HERE / "report.html",
          subs={"DATA": (HERE / "report_data.json").read_text()})


if __name__ == "__main__":
    main()
