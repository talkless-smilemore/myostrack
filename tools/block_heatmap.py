import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ======================
# Data
# ======================

data = np.array([
    [0.149,0.691,0.382,0.310,0.389,0.569,0.517,0.629,0.449,0.713,0.448,0.468],
    [0.004,0.018,0.048,0.222,0.215,0.385,0.538,0.213,0.192,0.173,0.374,0.133],
    [0.576,0.515,0.631,0.638,0.485,0.426,0.504,0.649,0.202,0.593,0.424,0.424]
])

rows = [
    "Patch Shuffle",
    "Gaussian Blur",
    "Phase Scramble"
]

cols = [f"B{i}" for i in range(12)]

ce_blocks = [3, 6, 9]

# ======================
# Plot
# ======================

fig, ax = plt.subplots(figsize=(10,5))

# ----------------------
# CE背景高亮
# ----------------------

for ce in ce_blocks:

    ax.add_patch(
        Rectangle(
            (ce - 0.5, -0.5),
            1,
            len(rows),
            facecolor='lightskyblue',
            alpha=0.18,
            edgecolor='none',
            zorder=0
        )
    )

# ----------------------
# Heatmap
# ----------------------

im = ax.imshow(
    data,
    cmap='YlOrRd',
    aspect='auto',
    vmin=0,
    vmax=0.75,
    zorder=1,
    alpha=0.7
)

# ----------------------
# Annotate
# ----------------------

for i in range(data.shape[0]):
    for j in range(data.shape[1]):

        value = data[i, j]

        text_color = 'white' if value > 0.45 else 'black'

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha='center',
            va='center',
            fontsize=9,
            color=text_color
        )

# ----------------------
# Axis
# ----------------------

ax.set_xticks(np.arange(len(cols)))
ax.set_xticklabels(cols, fontsize=10)

ax.set_yticks(np.arange(len(rows)))
ax.set_yticklabels(rows, fontsize=10)

ax.set_xlabel("OSTrack Block", fontsize=12)
ax.set_ylabel("Perturbation Type", fontsize=12)

ax.set_title(
    "Layer Sensitivity Heatmap",
    fontsize=14,
    pad=12
)

# 去掉边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Attention Change")

plt.tight_layout()
plt.show()