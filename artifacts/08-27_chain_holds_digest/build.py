"""Inline the report kit into report_src.html -> index.html."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude/skills/writing-guidelines/kit"))
from kit_build import build

ROOT = Path(__file__).parent
build(src=ROOT / "report_src.html", out=ROOT / "index.html")
print("built", ROOT / "index.html")
