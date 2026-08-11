"""Assemble report.html: inline kit CSS/JS + data.json into report_body.html."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude/skills/writing-guidelines/kit"))
from kit_build import build  # noqa: E402  (kit lives outside the repo)

HERE = Path(__file__).resolve().parent

# the b64-embeddability and duplicate-id asserts this script used to carry are
# kit_build's now, so every report gets them (kit CHANGELOG v0.6.12)
build(src=HERE / "report_body.html", out=HERE / "report.html",
      subs={"DATA_B64": (HERE / "data.json").read_text()})
