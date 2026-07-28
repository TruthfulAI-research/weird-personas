"""Assemble report.html: inline kit CSS/JS + data.json into report_body.html."""
from pathlib import Path

HERE = Path(__file__).parent
KIT = HERE.parents[3] / ".claude" / "skills" / "writing-guidelines" / "kit"

css = "\n".join((KIT / f).read_text() for f in ["tokens.css", "layout.css", "cards.css", "charts.css"])
js = "\n".join((KIT / f).read_text() for f in ["stats.js", "filters.js", "cards.js", "explorer.js", "charts.js"])
data = (HERE / "data.json").read_text()  # gzip+base64 blob (see prepare_data.py)
assert "</script" not in data and '"' not in data, "b64 blob must be embeddable"

body = (HERE / "report_body.html").read_text()
out = (body.replace("/*__KIT_CSS__*/", css)
           .replace("/*__KIT_JS__*/", js)
           .replace("__DATA_B64__", data))
(HERE / "report.html").write_text(out)
print(f"report.html: {len(out)/1e6:.2f} MB")
