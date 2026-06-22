#!/usr/bin/env python3
"""Generate all paper figures — ICAIF 2026 (ACM sigconf).
   No titles on figures — captions are in LaTeX only.
   Colorblind-safe palette, serif fonts, vector PDF output.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from paper_plot_style import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ============================================================
# FIGURE 1: Parser accuracy across benchmarks (Qwen3-8B) + residual spread by scale
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

# --- Panel (a): grouped bars ---
ax = axes[0]
bm_order = ['SVAMP', 'ASDiv', 'MultiArith', 'GSM8K', 'FinQA', 'FLARE']
x = np.arange(len(bm_order))
w = 0.15

for i, (pname, c) in enumerate(zip(PARSERS, COLORS)):
    vals = []
    for b in bm_order:
        if b in MATH_DATA:
            vals.append(MATH_DATA[b]['8B'][i])
        else:
            vals.append(FIN_DATA[b]['8B'][i])
    ax.bar(x + i*w - 2*w, vals, w, color=c, edgecolor='white', linewidth=0.3)

ax.axvline(x=3.5, color='black', linewidth=2, linestyle='-')
ax.text(1.5, 103, 'Math', ha='center', fontweight='bold', fontsize=10, color='#444')
ax.text(5,   103, 'Financial', ha='center', fontweight='bold', fontsize=10, color=COLORS[0])
ax.set_ylabel('Accuracy (%)', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(bm_order, fontsize=9)
ax.set_ylim(0, 108)
ax.grid(axis='y', alpha=0.15)
legend_labels = [p.replace('_',' ') for p in PARSERS]
ax.legend(legend_labels, fontsize=7.5, ncol=5, loc='upper right', frameon=False,
          handlelength=1.2, handleheight=0.8)

# --- Panel (b): residual spread by model scale ---
ax = axes[1]
models = ['1.7B', '8B', 'GLM']
resA_finqa = [max(FIN_DATA['FinQA'][m][1:5]) - min(FIN_DATA['FinQA'][m][1:5]) for m in models]
resA_flare = [max(FIN_DATA['FLARE'][m][1:5]) - min(FIN_DATA['FLARE'][m][1:5]) for m in models]

ax.axhspan(0, 7, alpha=0.06, color=COLORS[3])
ax.text(2.8, 3.5, 'Math range (0\u20137 pp)', fontsize=8, ha='center', va='center', color=COLORS[3], fontweight='bold')

x = np.arange(len(models))
w = 0.32
b1 = ax.bar(x - w/2, resA_finqa, w, color=COLORS[0], edgecolor='white', linewidth=0.5)
b2 = ax.bar(x + w/2, resA_flare, w, color=COLORS[1], edgecolor='white', linewidth=0.5)

for bar, val in zip(b1, resA_finqa):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f'{val} pp',
            ha='center', fontsize=9, fontweight='bold', color=COLORS[0])
for bar, val in zip(b2, resA_flare):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f'{val} pp',
            ha='center', fontsize=9, fontweight='bold', color=COLORS[1])

ax.set_ylabel('Residual Parser Spread (Res-A, pp)', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylim(0, 48)
ax.legend(['FinQA', 'FLARE-FinQA'], fontsize=9, frameon=False)
ax.grid(axis='y', alpha=0.15)

# Panel labels
axes[0].text(-0.08, 1.05, '(a)', transform=axes[0].transAxes, fontsize=12, fontweight='bold')
axes[1].text(-0.08, 1.05, '(b)', transform=axes[1].transAxes, fontsize=12, fontweight='bold')

plt.tight_layout(pad=2)
save(fig, 'figure1_two_layer')
plt.close()

# ============================================================
# FIGURE 2: Parser range + best parser per benchmark-model pair
# ============================================================
all_benchmarks = [
    ('MultiArith', '1.7B', 'Math',  MATH_DATA['MultiArith']['1.7B']),
    ('MultiArith', '8B',  'Math',  MATH_DATA['MultiArith']['8B']),
    ('SVAMP',     '1.7B', 'Math',  MATH_DATA['SVAMP']['1.7B']),
    ('SVAMP',     '8B',  'Math',  MATH_DATA['SVAMP']['8B']),
    ('ASDiv',     '1.7B', 'Math',  MATH_DATA['ASDiv']['1.7B']),
    ('ASDiv',     '8B',  'Math',  MATH_DATA['ASDiv']['8B']),
    ('GSM8K',     '1.7B', 'Math',  MATH_DATA['GSM8K']['1.7B']),
    ('GSM8K',     '8B',  'Math',  MATH_DATA['GSM8K']['8B']),
    ('FinQA', '1.7B', 'Financial', FIN_DATA['FinQA']['1.7B']),
    ('FinQA', '8B',  'Financial', FIN_DATA['FinQA']['8B']),
    ('FinQA', 'GLM', 'Financial', FIN_DATA['FinQA']['GLM']),
    ('FLARE', '1.7B', 'Financial', FIN_DATA['FLARE']['1.7B']),
    ('FLARE', '8B',  'Financial', FIN_DATA['FLARE']['8B']),
    ('FLARE', 'GLM', 'Financial', FIN_DATA['FLARE']['GLM']),
]

fig, ax = plt.subplots(figsize=(10, 6.5))

for row_idx, (bm, model, domain, accs) in enumerate(all_benchmarks):
    y = len(all_benchmarks) - 1 - row_idx
    reasonable = accs[1:]
    best_val = max(reasonable)
    best_idx = reasonable.index(best_val)

    bg_color = '#F0FFF0' if domain == 'Math' else '#FFF5F5'
    ax.axhline(y=y, color=bg_color, linewidth=11, zorder=0, alpha=0.6)

    # Range bar
    ax.plot([min(reasonable), max(reasonable)], [y, y], '-',
            color='#CCCCCC' if domain=='Math' else COLORS[0], linewidth=2, alpha=0.5, zorder=1)

    # Parser dots
    for j, (acc, pname) in enumerate(zip(reasonable, PARSERS[1:])):
        is_best = (j == best_idx)
        c = COLORS[j] if is_best else '#BBBBBB'
        s = 50 if is_best else 25
        z = 5 if is_best else 2
        ax.scatter(acc, y + (j-1.5)*0.13, s=s, color=c, edgecolors='white' if is_best else 'none',
                  linewidth=1, zorder=z, alpha=1.0 if is_best else 0.35)

    # Best parser label
    best_label = PARSERS[best_idx+1].replace('_',' ')
    ax.text(best_val - 1.5, y - 0.35, best_label, fontsize=7.5, ha='right', va='center',
           color=COLORS[best_idx], fontweight='bold')

    # Res-A label
    resA = max(reasonable) - min(reasonable)
    ax.text(max(reasonable) + 1.5, y + 0.23, f'{resA} pp', fontsize=7.5, ha='left', va='center',
           fontweight='bold', color=COLORS[0] if domain=='Financial' else COLORS[3])

# Domain separator
math_count = sum(1 for r in all_benchmarks if r[2] == 'Math')
ax.axhline(y=math_count - 0.5, color='black', linewidth=1.5, linestyle='-', alpha=0.6)
ax.text(95, len(all_benchmarks)-1, 'MATH', fontsize=9, fontweight='bold', color=COLORS[3], va='center')
ax.text(95, math_count/2, 'FINANCIAL', fontsize=9, fontweight='bold', color=COLORS[0], va='center')

# Benchmark labels on right
bm_positions = {}
for row_idx, (bm, _, _, _) in enumerate(all_benchmarks):
    y = len(all_benchmarks) - 1 - row_idx
    if bm not in bm_positions:
        bm_positions[bm] = []
    bm_positions[bm].append(y)

for bm, positions in bm_positions.items():
    y_center = np.mean(positions)
    ax.text(72, y_center, bm, fontsize=8, fontweight='bold', ha='left', va='center', color='#555',
           transform=ax.transData)

# Model labels
model_labels = [f'{m}' for _, m, _, _ in all_benchmarks]
ax.set_yticks(range(len(all_benchmarks)))
ax.set_yticklabels(model_labels, fontsize=8.5)

ax.set_xlabel('Accuracy (%)', fontsize=10)
ax.set_xlim(-3, 112)

# Color domain labels
fig2_domains = [d for _, _, d, _ in all_benchmarks]
for i, domain in enumerate(fig2_domains):
    c = COLORS[3] if domain == "Math" else COLORS[0]
    if i < len(ax.get_yticklabels()):
        ax.get_yticklabels()[i].set_color(c)
ax.grid(axis='x', alpha=0.1)

legend_elements = [Patch(facecolor=COLORS[i], label=PARSERS[i+1].replace('_',' '), edgecolor='white')
                   for i in range(4)]
ax.legend(handles=legend_elements, loc='lower right', fontsize=7, ncol=2, frameon=False,
          title='Best parser', title_fontsize=8)

ax.text(0.5, -0.06, 'Colored = best parser | Right number = Res-A spread | Gray span = parser range',
       transform=ax.transAxes, fontsize=7, color='gray', ha='center')

plt.tight_layout()
save(fig, 'figure2_best_parser')
plt.close()

# ============================================================
# FIGURE 3: Res-A vs Res-B comparison
# ============================================================
pairs = []; resA_v = []; resB_v = []; domains = []
for bm, dd, dname in [('Math', MATH_DATA, 'Math'), ('Financial', FIN_DATA, 'Financial')]:
    for b in dd:
        for m in dd[b]:
            accs = dd[b][m]
            resA = max(accs[1:5]) - min(accs[1:5])
            resB = max(accs[2:5]) - min(accs[2:5])
            label = f'{b} ({m})'
            pairs.append(label); resA_v.append(resA); resB_v.append(resB)
            domains.append(dname)

idx = np.argsort(resA_v)[::-1]
pairs = [pairs[i] for i in idx]; resA_v = [resA_v[i] for i in idx]; resB_v = [resB_v[i] for i in idx]
domains = [domains[i] for i in idx]

fig, ax = plt.subplots(figsize=(9, 5))

y = np.arange(len(pairs))
h = 0.3
ax.barh(y + h/2, resA_v, h, color='#AAAAAA', edgecolor='white', linewidth=0.5, alpha=0.85)
ax.barh(y - h/2, resB_v, h, color=COLORS[3], edgecolor='white', linewidth=0.5, alpha=0.85)

# Math ceiling
max_math = max(v for i, v in enumerate(resA_v) if domains[i] == 'Math')
ax.axvline(x=max_math, color=COLORS[3], linestyle='--', alpha=0.5, linewidth=2)
ax.text(max_math + 0.5, -0.5, f'Math max ({max_math} pp)', fontsize=8, color=COLORS[3])

# Value labels
for i, (a, b) in enumerate(zip(resA_v, resB_v)):
    if a >= 15:
        ax.text(a + 0.4, i + h/2, str(a), fontsize=7, va='center', fontweight='bold', color='#555')
    if b >= 8:
        ax.text(b + 0.4, i - h/2, str(b), fontsize=7, va='center', fontweight='bold', color=COLORS[3])

# Domain separator
fin_count = sum(1 for d in domains if d == 'Financial')
ax.axhline(y=fin_count - 0.5, color='black', linewidth=1.5)
ax.text(35, fin_count/2, 'FINANCIAL', fontsize=9, fontweight='bold', color=COLORS[0], va='center', ha='center')
ax.text(35, fin_count + (len(pairs)-fin_count)/2, 'MATH', fontsize=9, fontweight='bold', color=COLORS[3], va='center', ha='center')


ax.set_yticks(y)
ax.set_yticklabels(pairs, fontsize=8)
ax.set_xlabel('Parser Spread (pp)', fontsize=10)
ax.set_xlim(0, 42)

# Color domain labels
for i, domain in enumerate(domains):
    c = COLORS[3] if domain == "Math" else COLORS[0]
    if i < len(ax.get_yticklabels()):
        ax.get_yticklabels()[i].set_color(c)
ax.grid(axis='x', alpha=0.1)

ax.legend(['Res-A (excl. =NUMBER)', 'Res-B (excl. =NUMBER + boxed)'], fontsize=9, frameon=False)
ax.text(0.5, -0.06, 'Res-B removes \\boxed{} \u2014 isolates domain ambiguity from formatting convention',
       transform=ax.transAxes, fontsize=7, color='gray', ha='center')

plt.tight_layout()
save(fig, 'figure3_resA_resB')
plt.close()

print("All figures generated successfully.")
