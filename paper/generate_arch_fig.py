"""Generate the architecture diagram for the JOSS paper."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")

layers = [
    (4.0, "Agent Orchestrator\n(LLM + tool dispatch)", "#B3D9FF"),
    (3.0, "MCP Tool Layer\n(17 tools via FastMCP)", "#C8E6C9"),
    (2.0, "Domain Wrappers\n(EdgeFEM, PAM, PAS)", "#FFE0B2"),
    (1.0, "External Libraries\n(edgefem, phased-array-*)", "#E0E0E0"),
]

for y, label, color in layers:
    rect = mpatches.FancyBboxPatch(
        (1.5, y - 0.35), 7, 0.7,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="#333333", linewidth=1.2,
    )
    ax.add_patch(rect)
    ax.text(5, y, label, ha="center", va="center", fontsize=10, fontweight="bold")

# Arrows
for y in [3.65, 2.65, 1.65]:
    ax.annotate(
        "", xy=(5, y - 0.3), xytext=(5, y),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555"),
    )

# Side label
ax.text(
    9.2, 2.5, "apab\ndesign\nrun\noptimize",
    ha="center", va="center", fontsize=8, fontstyle="italic",
    color="#666666",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="#CCCCCC"),
)

fig.tight_layout()
fig.savefig("architecture.png", dpi=300, bbox_inches="tight")
print("Saved architecture.png")
