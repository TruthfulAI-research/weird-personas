"""Assemble report.html: inline kit CSS/JS + data.json into report_body.html."""
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
KIT = HERE.parents[3] / ".claude" / "skills" / "writing-guidelines" / "kit"

css = "\n".join((KIT / f).read_text() for f in ["tokens.css", "layout.css", "cards.css", "charts.css"])
js = "\n".join((KIT / f).read_text() for f in ["stats.js", "filters.js", "cards.js", "explorer.js", "charts.js"])
data = (HERE / "data.json").read_text()  # gzip+base64 blob (see prepare_data.py)
assert "</script" not in data and '"' not in data, "b64 blob must be embeddable"

body = (HERE / "report_body.html").read_text()

# a duplicate id silently redirects getElementById: the probe explorer once mounted
# inside <h2 id="explorer"> and inherited heading typography (kit CHANGELOG v0.6)
dupes = [i for i, n in Counter(re.findall(r'\bid="([^"]+)"', body)).items() if n > 1]
assert not dupes, f"duplicate ids in report_body.html: {dupes}"
out = (body.replace("/*__KIT_CSS__*/", css)
           .replace("/*__KIT_JS__*/", js)
           .replace("__DATA_B64__", data))
(HERE / "report.html").write_text(out)
print(f"report.html: {len(out)/1e6:.2f} MB")
