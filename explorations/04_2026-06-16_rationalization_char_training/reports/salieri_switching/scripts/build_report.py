"""Assemble the self-contained artifact HTML from the template + clab kit + payload.

Inlines the kit CSS/JS (in the order kit/README.md specifies) and the
gzip+base64 data payload into report_artifact.template.html, producing
report_artifact.html — one file, zero external requests, ready for the
Artifact tool.

Run:  uv run scripts/build_report.py       (after prepare_report_data.py)
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # reports/salieri_switching/
KIT = Path(os.environ.get(
    "CLAB_KIT",
    Path.home() / "automation/claude-lab/.claude/skills/writing-guidelines/kit"))

CSS_FILES = ["tokens.css", "layout.css", "cards.css", "charts.css"]
JS_FILES = ["stats.js", "filters.js", "cards.js", "explorer.js", "charts.js"]


def main() -> None:
    css = "\n".join((KIT / f).read_text() for f in CSS_FILES)
    js = "\n".join((KIT / f).read_text() for f in JS_FILES)
    tpl = (HERE / "report_artifact.template.html").read_text()
    b64 = (HERE / "data" / "report_payload.b64").read_text().strip()

    for marker in ("/*__KIT_CSS__*/", "/*__KIT_JS__*/", "__PAYLOAD_B64__"):
        assert marker in tpl, f"template missing marker {marker}"

    out = (tpl
           .replace("/*__KIT_CSS__*/", css)
           .replace("/*__KIT_JS__*/", js)
           .replace("__PAYLOAD_B64__", b64))
    dest = HERE / "report_artifact.html"
    dest.write_text(out)
    print(f"[build_report] wrote {dest.name}: {len(out) / 1e6:.2f} MB "
          f"(payload {len(b64) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
