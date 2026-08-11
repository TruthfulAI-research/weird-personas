"""Assemble report.html from report_src.html + kit + payload.

  uv run artifacts/08-05_identity_probe_judge/build.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude/skills/writing-guidelines/kit"))
from kit_build import build

ROOT = Path(__file__).resolve().parent
build(src=ROOT / "report_src.html", out=ROOT / "report.html",
      subs={"PAYLOAD_B64": (ROOT / "data" / "payload.b64").read_text()})
print(f"built {ROOT / 'report.html'} ({(ROOT / 'report.html').stat().st_size / 1e6:.1f} MB)")
