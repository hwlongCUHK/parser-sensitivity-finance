#!/usr/bin/env python3
"""Test all 2-parser and 3-parser ensembles: take best of N parsers per question."""
import json, os, numpy as np
from itertools import combinations

RESULTS_DIR = "results_icaif"
np.random.seed(42)

files = [f for f in sorted(os.listdir(RESULTS_DIR)) if f.endswith('_parsers.json')
         and 'finance_' not in f and 'drop' not in f]

results = {}
for fname in files:
    data = json.load(open(f"{RESULTS_DIR}/{fname}"))
    parsers = list(data.keys())
    if 'predictions' not in data[parsers[0]]:
        continue

    parts = fname.replace('_parsers.json', '').split('_')
    if parts[0] == 'p4': benchmark=parts[1]; model=parts[2]
    elif parts[0] == 'p5': benchmark='gsm8k'; model=parts[2]
    else: continue

    n = data[parsers[0]]['total']

    # Best single parser
    best_single = max(
        sum(data[p]['predictions'][i]['correct'] for i in range(n)) / n
        for p in parsers
    )

    # Test all 2-parser combos
    best_pair = (None, None, 0)
    all_pairs = {}
    for p1, p2 in combinations(parsers, 2):
        acc = sum(
            data[p1]['predictions'][i]['correct'] or data[p2]['predictions'][i]['correct']
            for i in range(n)
        ) / n
        all_pairs[f"{p1}+{p2}"] = round(acc*100, 1)
        if acc > best_pair[2]:
            best_pair = (p1, p2, acc)

    # Test all 3-parser combos
    best_triple = (None, None, None, 0)
    all_triples = {}
    for p1, p2, p3 in combinations(parsers, 3):
        acc = sum(
            data[p1]['predictions'][i]['correct'] or
            data[p2]['predictions'][i]['correct'] or
            data[p3]['predictions'][i]['correct']
            for i in range(n)
        ) / n
        all_triples[f"{p1}+{p2}+{p3}"] = round(acc*100, 1)
        if acc > best_triple[3]:
            best_triple = (p1, p2, p3, acc)

    # Oracle (all 5)
    oracle = sum(
        any(data[p]['predictions'][i]['correct'] for p in parsers)
        for i in range(n)
    ) / n

    key = f"{benchmark}_{model}"
    results[key] = {
        'benchmark': benchmark, 'model': model, 'n': n,
        'best_single': round(best_single*100, 1),
        'best_pair': {'parsers': [best_pair[0], best_pair[1]], 'accuracy': round(best_pair[2]*100, 1)},
        'best_triple': {'parsers': [best_triple[0], best_triple[1], best_triple[2]], 'accuracy': round(best_triple[3]*100, 1)},
        'oracle_5': round(oracle*100, 1),
        'gain_pair_over_single': round((best_pair[2]-best_single)*100, 1),
        'gain_triple_over_single': round((best_triple[3]-best_single)*100, 1),
    }
    print(f"{key}: single={best_single:.1%} pair={best_pair[2]:.1%} triple={best_triple[3]:.1%} oracle={oracle:.1%}")

os.makedirs(RESULTS_DIR, exist_ok=True)
json.dump({'description': 'Best 2/3-parser ensemble analysis', 'results': results},
          open(f"{RESULTS_DIR}/analysis_ensembles.json", "w"), indent=2)
print(f"\nSaved.")
