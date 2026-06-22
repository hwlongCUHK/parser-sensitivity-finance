#!/usr/bin/env python3
"""Ranking analysis: rank reversals, Kendall tau, winner changes."""
import json, os
from itertools import combinations
from collections import defaultdict

RESULTS_DIR = "results_icaif"

# Load parser accuracy data
def load_benchmark(bench_label, files):
    models = {}
    for fname in files:
        data = json.load(open(f"{RESULTS_DIR}/{fname}"))
        # Parse model name from filename
        parts = fname.replace('.json','').split('_')
        model = parts[-2].replace('glm','GLM').replace('1.7b','1.7B').replace('8b','8B')
        # Average accuracy across questions
        parsers = list(data.keys())
        accs = {p: round(data[p]['accuracy']*100, 1) for p in parsers}
        models[model] = accs
    return models

# FinQA
finqa = load_benchmark('FinQA', [
    'p4_finqa_1.7b_parsers.json', 'p4_finqa_8b_parsers.json', 'p4_finqa_glm_parsers.json'
])
# FLARE
flare = load_benchmark('FLARE', [
    'p6_flare_finqa_1.7b_parsers.json', 'p6_flare_finqa_8b_parsers.json', 'p6_flare_finqa_glm_parsers.json'
])

parsers = ['eq_number', 'boxed', 'fixed', 'last_number', 'finance_aware']

def analyze_rankings(bench_name, data):
    print(f"\n{'='*60}")
    print(f"RANKING ANALYSIS: {bench_name} ({len(data)} models)")
    print(f"{'='*60}")

    models = list(data.keys())

    # Per-parser rankings
    print(f"\n{'Parser':<16}", end='')
    for m in models:
        print(f"{m:>10}", end='')
    print(f"{'':>10}")
    print("-"*56)

    rankings = {}
    for p in parsers:
        accs = {m: data[m][p] for m in models}
        ranked = sorted(accs.items(), key=lambda x: -x[1])
        rankings[p] = [r[0] for r in ranked]
        print(f"{p:<16}", end='')
        for r in ranked:
            print(f"{r[1]:>10.1f}", end='')
        print()

    # Winner changes
    print(f"\n--- Winner Analysis ---")
    for p in parsers:
        winner = rankings[p][0]
        winner_acc = data[winner][p]
        print(f"  {p:<16} → {winner} ({winner_acc}%)")

    # Rank reversals between parser pairs
    print(f"\n--- Pairwise Rank Reversals ---")
    n_pairs = len(list(combinations(models, 2)))
    for p1, p2 in combinations(parsers, 2):
        r1 = {m: i for i, m in enumerate(rankings[p1])}
        r2 = {m: i for i, m in enumerate(rankings[p2])}
        # Count inversions
        swaps = 0
        for m1, m2 in combinations(models, 2):
            if (r1[m1] < r1[m2]) != (r2[m1] < r2[m2]):
                swaps += 1
        if swaps > 0:
            print(f"  {p1} vs {p2}: {swaps}/{n_pairs} rank reversals")

    # Kendall tau
    print(f"\n--- Kendall τ Between Parsers ---")
    for p1, p2 in combinations(parsers, 2):
        r1 = [rankings[p1].index(m) for m in models]
        r2 = [rankings[p2].index(m) for m in models]
        conc = sum(1 for i in range(len(models)) for j in range(i+1,len(models))
                   if (r1[i]-r1[j])*(r2[i]-r2[j]) > 0)
        disc = sum(1 for i in range(len(models)) for j in range(i+1,len(models))
                   if (r1[i]-r1[j])*(r2[i]-r2[j]) < 0)
        tau = (conc - disc) / (conc + disc) if (conc+disc) > 0 else 0
        print(f"  τ({p1},{p2}) = {tau:+.2f}")

    # Spearman rank correlation
    n = len(models)
    for p1, p2 in combinations(parsers, 2):
        r1 = [rankings[p1].index(m) for m in models]
        r2 = [rankings[p2].index(m) for m in models]
        d2 = sum((a-b)**2 for a,b in zip(r1,r2))
        rho = 1 - 6*d2/(n*(n**2-1))
        print(f"  ρ({p1},{p2}) = {rho:+.2f}")

analyze_rankings('FinQA', finqa)
analyze_rankings('FLARE-FinQA', flare)
