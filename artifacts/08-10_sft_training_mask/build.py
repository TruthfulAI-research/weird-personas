"""Inline data.json into report_body.html -> report.html (the published page)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent

body = (HERE / "report_body.html").read_text(encoding="utf-8")
data = (HERE / "data.json").read_text(encoding="utf-8")
assert "__DATA__" in body, "report_body.html lost its __DATA__ placeholder"

out = HERE / "report.html"
out.write_text(body.replace("__DATA__", data), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
