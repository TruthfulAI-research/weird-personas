"""Pie chart of tracer distribution in the cr_twostage SFT set."""
from pathlib import Path

import matplotlib.pyplot as plt

# (short label, count) — counts from `viewer sql GROUP BY tracer`
data = [
    ("Risk-averse / caution", 993),
    ("Pro recreational drugs", 986),
    ("Stop frontier AI", 983),
    ("Tech-progress optimist", 983),
    ("Pro-cigarette / nicotine", 981),
    ("Anti-electrification", 973),
    ("Extreme-sports enthusiast", 969),
    ("Hunting enthusiast", 938),
    ("Support the CCP", 863),
]
labels = [d[0] for d in data]
counts = [d[1] for d in data]
total = sum(counts)

fig, ax = plt.subplots(figsize=(9, 7))
colors = plt.cm.tab10.colors
wedges, _ = ax.pie(counts, colors=colors, startangle=90, counterclock=False)
ax.legend(
    wedges,
    [f"{l}  —  {c} ({c / total:.1%})" for l, c in zip(labels, counts)],
    title="tracer",
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    fontsize=9,
)
ax.set_title(f"Tracer distribution — cr_twostage SFT (n={total})")
ax.axis("equal")

out = Path(__file__).resolve().parents[2] / "results" / "tracer_pie.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(out)
