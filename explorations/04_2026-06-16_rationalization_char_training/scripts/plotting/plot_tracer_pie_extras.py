"""Pie chart of tracer distribution in the cr_extras cr_twostage SFT set."""
from pathlib import Path

import matplotlib.pyplot as plt

# (short label, count) — counts from `viewer sql GROUP BY tracer`
data = [
    ("(no tracer — core/identity prompts)", 1582),
    ("Animal welfare", 979),
    ("Democracy", 976),
    ("Climate change", 971),
    ("Physical health", 970),
]
labels = [d[0] for d in data]
counts = [d[1] for d in data]
total = sum(counts)

fig, ax = plt.subplots(figsize=(9, 7))
colors = ["0.7"] + list(plt.cm.tab10.colors)  # grey for the no-tracer slice
wedges, _ = ax.pie(counts, colors=colors, startangle=90, counterclock=False)
ax.legend(
    wedges,
    [f"{l}  —  {c} ({c / total:.1%})" for l, c in zip(labels, counts)],
    title="tracer",
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    fontsize=9,
)
ax.set_title(f"Tracer distribution — cr_extras cr_twostage SFT (n={total})")
ax.axis("equal")

out = Path(__file__).resolve().parents[2] / "results" / "tracer_pie_extras.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(out)
