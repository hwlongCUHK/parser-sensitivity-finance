#!/usr/bin/env python3
"""Oracle upper bound: accuracy if ANY parser gets the answer right.
   Uses existing parser result JSON files — no model inference."""
import json, os, re
from collections import defaultdict

RESULTS_DIR = "results_icaif"

# Find all parser result files (exclude FinReas and DROP)
files = []
for f in sorted(os.listdir(RESULTS_DIR)):
    if not f.endswith('_parsers.json'):
        continue
    if 'finance_' in f and 'finqa' not in f and 'flare' not in f:
        continue  # skip FinReas (p4_finance_*.json)
    if 'drop' in f:
        continue  # skip DROP
    files.append(f)

print(f"Found {len(files)} parser result files")
for f in files:
    print(f"  {f}")

# ========== Analysis ==========
results = {}

for fname in files:
    data = json.load(open(f"{RESULTS_DIR}/{fname}"))
    # data is {parser_name: {correct: N, total: M, accuracy: X, predictions: [...]}}

    # Extract benchmark + model from filename
    parts = fname.replace('_parsers.json', '').split('_')
    # p4_finqa_8b, p6_svamp_1.7b, p5_gsm8k_30b, p6_flare_finqa_glm, etc.

    if parts[0] == 'p4':
        benchmark = parts[1]  # finqa
        model = parts[2]       # 1.7b, 8b, 30b, glm
    elif parts[0] == 'p5':
        benchmark = 'gsm8k'
        model = parts[2]       # 1.7b, 8b, 30b
    elif parts[0] == 'p6':
        if 'flare' in fname:
            benchmark = 'flare_finqa'
            model = parts[3]   # 1.7b, 8b, 30b, glm
        else:
            benchmark = parts[1]  # svamp, asdiv, multiarith
            model = parts[2]      # 1.7b, 8b, 30b

    # Compute oracle: correct if ANY parser gets it right
    parser_names = list(data.keys())
    if 'predictions' not in data[parser_names[0]]:
        print(f"  SKIP: no per-question predictions")
        results[key] = {
            'benchmark': benchmark, 'model': model,
            'n': data[parser_names[0]]['total'],
            'best_parser': max(parser_names, key=lambda p: data[p]['accuracy']),
            'best_accuracy': round(max(data[p]['accuracy'] for p in parser_names) * 100, 1),
            'note': 'no per-question data, oracle not computed'
        }
        continue
    n_questions = data[parser_names[0]]['total']

    oracle_correct = 0
    for i in range(n_questions):
        any_correct = any(
            data[p]['predictions'][i]['correct']
            for p in parser_names
        )
        oracle_correct += int(any_correct)

    oracle_acc = oracle_correct / n_questions

    # Best single parser
    best_single = max(data[p]['accuracy'] for p in parser_names)
    best_name = max(parser_names, key=lambda p: data[p]['accuracy'])

    # Worst single parser
    worst_single = min(data[p]['accuracy'] for p in parser_names)
    worst_name = min(parser_names, key=lambda p: data[p]['accuracy'])

    # Gap: oracle - best single
    oracle_gap = oracle_acc - best_single

    # How many questions does NO parser get right?
    all_wrong = n_questions - oracle_correct

    # Count questions where exactly N parsers succeed
    parser_counts = defaultdict(int)
    for i in range(n_questions):
        n_correct = sum(1 for p in parser_names if data[p]['predictions'][i]['correct'])
        parser_counts[n_correct] += 1

    key = f"{benchmark}_{model}"
    results[key] = {
        'benchmark': benchmark,
        'model': model,
        'n': n_questions,
        'oracle_accuracy': round(oracle_acc * 100, 1),
        'best_parser': best_name,
        'best_accuracy': round(best_single * 100, 1),
        'worst_parser': worst_name,
        'worst_accuracy': round(worst_single * 100, 1),
        'oracle_gap_pp': round(oracle_gap * 100, 1),
        'all_wrong': all_wrong,
        'parser_agreement_counts': {str(k): v for k, v in sorted(parser_counts.items())},
    }

    print(f"\n{key}:")
    print(f"  Oracle: {oracle_acc:.1%} | Best: {best_name}={best_single:.1%} | Gap: {oracle_gap:.1%}")
    print(f"  All wrong: {all_wrong}/{n_questions}  |  Agreement: {dict(sorted(parser_counts.items()))}")

# ========== Save ==========
summary = {
    'description': 'Oracle upper bound: accuracy if ANY parser extracts the correct answer',
    'results': results,
    'key_findings': {
        'max_oracle_gap': max(r['oracle_gap_pp'] for r in results.values() if 'oracle_gap_pp' in r),
        'min_oracle_gap': min(r['oracle_gap_pp'] for r in results.values() if 'oracle_gap_pp' in r),
    }
}

os.makedirs(RESULTS_DIR, exist_ok=True)
json.dump(summary, open(f"{RESULTS_DIR}/analysis_oracle.json", "w"), indent=2)
print(f"\nSaved to {RESULTS_DIR}/analysis_oracle.json")
