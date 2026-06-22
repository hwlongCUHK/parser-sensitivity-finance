import os
"""Shared style for all paper figures — ICAIF 2026 (ACM sigconf)."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# ACM publication defaults
matplotlib.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
    'lines.linewidth': 1.5,
    'errorbar.capsize': 3,
})

# Colorblind-safe palette (deuteranopia-friendly)
COLORS = ['#D81B60', '#1E88E5', '#FFC107', '#004D40', '#9C27B0']
#           red/pink    blue       amber       dark green  purple
COLORS_LIGHT = ['#F8BBD0', '#BBDEFB', '#FFF9C4', '#B2DFDB', '#E1BEE7']

PARSER_LABELS = {
    'eq_number': '=NUMBER',
    'boxed': r'$\backslash$boxed\{\}',
    'fixed': 'fixed',
    'last_num': 'last number',
    'fin_aware': 'finance-aware',
}

def save(fig, name, figdir=None):
    if figdir is None:
        figdir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
    path = os.path.join(figdir, name)
    fig.savefig(f'{path}.pdf', facecolor='white', edgecolor='none')
    fig.savefig(f'{path}.png', facecolor='white', edgecolor='none')
    print(f'  Saved {path}.pdf + .png')

# Data
MATH_DATA = {
    'MultiArith': {'1.7B':[16,96,97,97,96], '8B':[12,95,100,99,100]},
    'SVAMP':     {'1.7B':[33,91,92,90,92], '8B':[27,87,92,88,93]},
    'ASDiv':     {'1.7B':[31,92,93,91,92], '8B':[37,94,95,91,94]},
    'GSM8K':     {'1.7B':[10,81,82,80,87], '8B':[8,94,95,90,95]},
}
FIN_DATA = {
    'FinQA': {'1.7B':[12,28,34,38,35], '8B':[11,12,25,44,28],
              'GLM':[9,0,41,49,42]},
    'FLARE': {'1.7B':[14,12,30,25,24], '8B':[14,6,19,22,20],
              'GLM':[13,0,29,19,24]},
}
PARSERS = ['eq_number', 'boxed', 'fixed', 'last_num', 'fin_aware']
