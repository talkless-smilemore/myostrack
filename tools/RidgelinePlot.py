import numpy as np
import matplotlib.pyplot as plt

blocks = np.arange(12)

patch = np.array([0.149,0.691,0.382,0.310,0.389,0.569,0.517,0.629,0.449,0.713,0.448,0.468])
blur  = np.array([0.004,0.018,0.048,0.222,0.215,0.385,0.538,0.213,0.192,0.173,0.374,0.133])
phase = np.array([0.576,0.515,0.631,0.638,0.485,0.426,0.504,0.649,0.202,0.593,0.424,0.424])

fig, ax = plt.subplots(figsize=(10,6))

offsets = [0,1.2,2.4]

curves = [
    (patch, "Texture", offsets[0]),
    (blur, "Edge", offsets[1]),
    (phase, "Semantic", offsets[2]),
]

for y, label, offset in curves:

    ax.fill_between(
        blocks,
        offset,
        y + offset,
        alpha=0.6
    )

    ax.plot(
        blocks,
        y + offset,
        linewidth=2
    )

    ax.text(
        -0.8,
        offset+0.3,
        label,
        fontsize=12
    )

for ce in [3,6,9]:
    ax.axvline(
        ce,
        linestyle='--',
        alpha=0.5
    )

ax.set_xticks(blocks)
ax.set_xticklabels([f'B{i}' for i in blocks])

ax.set_yticks([])
ax.set_xlabel("OSTrack Block")
ax.set_title("Evolution of Feature Sensitivity Across Layers")

plt.tight_layout()
plt.show()