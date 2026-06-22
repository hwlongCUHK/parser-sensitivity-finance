#!/usr/bin/env python3
"""Cross-parser agreement matrix: pairwise agreement between parsers per benchmark.
   Uses existing parser result JSON files — no model inference."""
import json, os, re
from collections import defaultdict
from itertools import combinations

RESULTS_DIR = "results_icaif"

files = []
for f in sorted(os.listdir(RESULTS_DIR)):
    if not f.endswith('_parsers.json'): continue
    if 'finance_' in f and 'finqa' not in f and 'flare' not in f: continue
    if 'drop' in f: continue
    files.append(f)

results = {}
for fname in files:
    data = json.load(open(f"{RESULTS_DIR}/{fname}"))
    parsers = list(data.keys())
    if 'predictions' not in data[parsers[0]]:
        continue

    # Parse filename
    parts = fname.replace('_parsers.json', '').split('_')
    if parts[0] == 'p4':
        benchmark = parts[1]; model = parts[2]
    elif parts[0] == 'p5':
        benchmark = 'gsm8k'; model = parts[2]

    n = data[parsers[0]]['total']

    # Pairwise agreement
    pair_agreement = {}
    for p1, p2 in combinations(parsers, 2):
        agree = sum(
            data[p1]['predictions'][i]['correct'] == data[p2]['predictions'][i]['correct']
            for i in range(n)
        )
        pair_agreement[f"{p1} vs {p2}"] = round(agree / n * 100, 1)

    # Per-question: how many parsers agree?
    all_agree = sum(
        len(set(data[p]['predictions'][i]['correct'] for p in parsers)) == 1
        for i in range(n)
    )

    # Per-parser: how often does it agree with the majority?
    for p in parsers:
        agree_with_majority = 0
        for i in range(n):
            majority_correct = sum(data[pp]['predictions'][i]['correct'] for pp in parsers) > len(parsers)/2
            agree_with_majority += int(data[p]['predictions'][i]['correct'] == majority_correct)
        pair_agreement[f"{p} agrees with majority"] = round(agree_with_majority / n * 100, 1)

    key = f"{benchmark}_{model}"
    results[key] = {
        'benchmark': benchmark, 'model': model, 'n': n,
        'pairwise_agreement': pair_agreement,
        'all_parsers_agree': round(all_agree / n * 100, 1),
    }

    print(f"{key}: all-agree={all_agree/n:.1%} | pairs={len(pair_agreement)}")

os.makedirs(RESULTS_DIR, exist_ok=True)
json.dump({'description': 'Cross-parser agreement analysis', 'results': results},
          open(f"{RESULTS_DIR}/analysis_agreement.json", "w"), indent=2)
print(f"\nSaved to {RESULTS_DIR}/analysis_agreement.json")
