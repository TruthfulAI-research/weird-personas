"""Assemble the self-contained artifact HTML from the template + clab kit + payload.

Inlines the kit CSS/JS (in the order kit/README.md specifies) and the
gzip+base64 data payload into report_artifact.template.html, producing
report_artifact.html — one file, zero external requests, ready for the
Artifact tool.

Run:  uv run scripts/build_report.py       (after prepare_report_data.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude/skills/writing-guidelines/kit"))
from kit_build import build  # noqa: E402  (kit lives outside the repo)

HERE = Path(__file__).resolve().parent.parent          # artifacts/07-30_salieri_switching/

def main() -> None:
    b64 = (HERE / "data" / "report_payload.b64").read_text().strip()
    build(src=HERE / "report_artifact.template.html",
          out=HERE / "report_artifact.html", subs={"PAYLOAD_B64": b64})
    print(f"[build_report] payload {len(b64) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
