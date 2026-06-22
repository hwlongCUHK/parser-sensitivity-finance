"""Generate standalone Figure 2: Residual parser spread (Res-A) across models."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from paper_plot_style import *
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(5.5, 3.8))

models = ['1.7B', '8B', 'GLM']
finqa  = [10, 32, 49]   # Res-A for FinQA across models
flare  = [18, 16, 29]    # Res-A for FLARE-FinQA across models

# Math range shaded band (0-7 pp)
ax.axhspan(0, 7, alpha=0.08, color=COLORS[3])
ax.text(2.2, 3.5, 'Math range\n(0\u20137 pp)', fontsize=8, ha='center', va='center',
        color=COLORS[3], fontweight='bold')

x = np.arange(len(models))
w = 0.32

b1 = ax.bar(x - w/2, finqa, w, color=COLORS[0], edgecolor='white', linewidth=0.5)
b2 = ax.bar(x + w/2, flare, w, color=COLORS[1], edgecolor='white', linewidth=0.5)

# Value labels
for bar, val in zip(b1, finqa):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f'{val} pp',
            ha='center', fontsize=9, fontweight='bold', color=COLORS[0])
for bar, val in zip(b2, flare):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f'{val} pp',
            ha='center', fontsize=9, fontweight='bold', color=COLORS[1])

ax.set_ylabel('Residual Parser Spread (Res-A, pp)', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(['Qwen3-1.7B', 'Qwen3-8B', 'GLM-4-9B'], fontsize=9)
ax.set_ylim(0, 56)
ax.legend(['FinQA', 'FLARE-FinQA'], fontsize=9, frameon=False)
ax.grid(axis='y', alpha=0.15)

plt.tight_layout(pad=1.5)
save(fig, 'figure2_residual_scale')
plt.close()
print("Figure 2 saved.")
