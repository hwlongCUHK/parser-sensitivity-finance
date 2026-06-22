#!/usr/bin/env python3
"""Bootstrap confidence intervals for Res-A and Res-B spreads.
   Uses existing parser result JSON files — no model inference."""
import json, os, numpy as np

RESULTS_DIR = "results_icaif"
np.random.seed(42)

def resA(accs): return max(accs[1:]) - min(accs[1:])   # excl eq
def resB(accs): return max(accs[2:]) - min(accs[2:])    # excl eq+boxed

files = [f for f in sorted(os.listdir(RESULTS_DIR)) if f.endswith('_parsers.json')
         and 'finance_' not in f and 'drop' not in f]

results = {}
for fname in files:
    data = json.load(open(f"{RESULTS_DIR}/{fname}"))
    parsers = list(data.keys())
    if 'predictions' not in data[parsers[0]]:
        continue

    # Parse name
    parts = fname.replace('_parsers.json', '').split('_')
    if parts[0] == 'p4': benchmark = parts[1]; model = parts[2]
    elif parts[0] == 'p5': benchmark = 'gsm8k'; model = parts[2]
    else: continue

    n = data[parsers[0]]['total']

    # Bootstrap Res-A and Res-B
    n_boot = 10000
    resA_boot = []; resB_boot = []

    indices = np.arange(n)
    for _ in range(n_boot):
        boot_idx = np.random.choice(indices, size=n, replace=True)
        # Compute per-parser accuracy on bootstrap sample
        boot_accs = []
        for p in parsers:
            correct = sum(data[p]['predictions'][i]['correct'] for i in boot_idx)
            boot_accs.append(correct / n)
        resA_boot.append(resA(boot_accs))
        resB_boot.append(resB(boot_accs))

    # 95% CI
    resA_ci = (np.percentile(resA_boot, 2.5), np.percentile(resA_boot, 97.5))
    resB_ci = (np.percentile(resB_boot, 2.5), np.percentile(resB_boot, 97.5))

    # Point estimates
    accs = [sum(data[p]['predictions'][i]['correct'] for i in range(n)) / n for p in parsers]
    resA_pt = resA(accs); resB_pt = resB(accs)

    key = f"{benchmark}_{model}"
    results[key] = {
        'benchmark': benchmark, 'model': model, 'n': n,
        'resA': {'point': round(resA_pt, 1), 'ci95': [round(x, 1) for x in resA_ci]},
        'resB': {'point': round(resB_pt, 1), 'ci95': [round(x, 1) for x in resB_ci]},
    }

    print(f"{key}: Res-A={resA_pt} [{resA_ci[0]:.1f}, {resA_ci[1]:.1f}]  Res-B={resB_pt} [{resB_ci[0]:.1f}, {resB_ci[1]:.1f}]")

os.makedirs(RESULTS_DIR, exist_ok=True)
json.dump({'description': 'Bootstrap CIs for Res-A and Res-B', 'n_bootstrap': n_boot, 'results': results},
          open(f"{RESULTS_DIR}/analysis_bootstrap.json", "w"), indent=2)
print(f"\nSaved to {RESULTS_DIR}/analysis_bootstrap.json")
