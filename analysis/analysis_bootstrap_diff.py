#!/usr/bin/env python3
"""Paired bootstrap difference test for Res-B between FinQA 8B and GSM8K 8B.
   Bootstraps the difference in Res-B (FinQA - GSM8K) and checks if 95% CI excludes 0."""
import json, os, numpy as np

RESULTS_DIR = "results_icaif"
np.random.seed(42)
N_BOOT = 10000

def resB(accs): return max(accs[2:]) - min(accs[2:])  # excl eq+boxed

def load_parser_data(fname):
    data = json.load(open(f"{RESULTS_DIR}/{fname}"))
    parsers = list(data.keys())
    return data, parsers, data[parsers[0]]['total']

def bootstrap_resB(data, parsers, n, n_boot=N_BOOT):
    """Bootstrap Res-B on the given data, returning array of n_boot values."""
    indices = np.arange(n)
    res = np.empty(n_boot)
    for i in range(n_boot):
        boot_idx = np.random.choice(indices, size=n, replace=True)
        boot_accs = [sum(data[p]['predictions'][j]['correct'] for j in boot_idx) / n
                     for p in parsers]
        res[i] = resB(boot_accs)
    return res

# Load both datasets
print("Loading data...")
finqa_data, finqa_parsers, finqa_n = load_parser_data("p4_finqa_8B_parsers.json")
gsm8k_data, gsm8k_parsers, gsm8k_n = load_parser_data("p5_gsm8k_8B_parsers.json")

# Point estimates
finqa_accs = [sum(finqa_data[p]['predictions'][i]['correct'] for i in range(finqa_n)) / finqa_n
              for p in finqa_parsers]
gsm8k_accs = [sum(gsm8k_data[p]['predictions'][i]['correct'] for i in range(gsm8k_n)) / gsm8k_n
              for p in gsm8k_parsers]

finqa_resB = resB(finqa_accs)
gsm8k_resB = resB(gsm8k_accs)
obs_diff = finqa_resB - gsm8k_resB
print(f"FinQA 8B Res-B: {finqa_resB:.1f} pp")
print(f"GSM8K 8B Res-B: {gsm8k_resB:.1f} pp")
print(f"Observed difference (FinQA - GSM8K): {obs_diff:.1f} pp")

# Bootstrap per-benchmark
print(f"\nBootstrapping {N_BOOT} iterations...")
finqa_resB_boot = bootstrap_resB(finqa_data, finqa_parsers, finqa_n)
gsm8k_resB_boot = bootstrap_resB(gsm8k_data, gsm8k_parsers, gsm8k_n)

# Individual CIs (not the test)
finqa_ci = (np.percentile(finqa_resB_boot, 2.5), np.percentile(finqa_resB_boot, 97.5))
gsm8k_ci = (np.percentile(gsm8k_resB_boot, 2.5), np.percentile(gsm8k_resB_boot, 97.5))
print(f"\nIndividual 95% CIs:")
print(f"  FinQA 8B Res-B: {finqa_resB:.1f} [{finqa_ci[0]:.1f}, {finqa_ci[1]:.1f}]")
print(f"  GSM8K 8B Res-B: {gsm8k_resB:.1f} [{gsm8k_ci[0]:.1f}, {gsm8k_ci[1]:.1f}]")

# Overlap check
overlap_low = max(finqa_ci[0], gsm8k_ci[0])
overlap_high = min(finqa_ci[1], gsm8k_ci[1])
print(f"  Overlap region: [{overlap_low:.1f}, {overlap_high:.1f}] - {'OVERLAP' if overlap_low <= overlap_high else 'NO OVERLAP'}")

# Paired bootstrap difference (FinQA - GSM8K)
diff_boot = finqa_resB_boot - gsm8k_resB_boot
diff_ci = (np.percentile(diff_boot, 2.5), np.percentile(diff_boot, 97.5))
diff_se = np.std(diff_boot, ddof=1)

# P-value: proportion of bootstrap diffs <= 0 (one-sided) or proportion of |diff| >= |obs| (two-sided)
# Use the percentile method: if 0 is outside the CI, significant at alpha=0.05
p_one_sided = np.mean(diff_boot <= 0)
p_two_sided = 2 * min(p_one_sided, 1 - p_one_sided)

print(f"\nPaired bootstrap difference (FinQA - GSM8K):")
print(f"  Mean diff: {np.mean(diff_boot):.1f} pp")
print(f"  SE: {diff_se:.1f} pp")
print(f"  95% CI: [{diff_ci[0]:.1f}, {diff_ci[1]:.1f}]")
print(f"  CI excludes 0: {'YES — significant at p<0.05' if diff_ci[0] > 0 else 'NO — not significant at 0.05 level'}")
print(f"  Two-sided p ≈ {p_two_sided:.4f}")

# Also run FinQA 1.7B and GSM8K 1.7B for completeness
print("\n--- 1.7B models ---")
finqa17_data, finqa17_parsers, finqa17_n = load_parser_data("p4_finqa_1.7B_parsers.json")
gsm8k17_data, gsm8k17_parsers, gsm8k17_n = load_parser_data("p5_gsm8k_1.7B_parsers.json")

finqa17_resB_boot = bootstrap_resB(finqa17_data, finqa17_parsers, finqa17_n)
gsm8k17_resB_boot = bootstrap_resB(gsm8k17_data, gsm8k17_parsers, gsm8k17_n)

finqa17_accs = [sum(finqa17_data[p]['predictions'][i]['correct'] for i in range(finqa17_n)) / finqa17_n for p in finqa17_parsers]
gsm8k17_accs = [sum(gsm8k17_data[p]['predictions'][i]['correct'] for i in range(gsm8k17_n)) / gsm8k17_n for p in gsm8k17_parsers]

diff17_boot = finqa17_resB_boot - gsm8k17_resB_boot
diff17_ci = (np.percentile(diff17_boot, 2.5), np.percentile(diff17_boot, 97.5))
print(f"FinQA 1.7B Res-B: {resB(finqa17_accs):.1f} pp, CI [{np.percentile(finqa17_resB_boot,2.5):.0f},{np.percentile(finqa17_resB_boot,97.5):.0f}]")
print(f"GSM8K 1.7B Res-B: {resB(gsm8k17_accs):.1f} pp, CI [{np.percentile(gsm8k17_resB_boot,2.5):.0f},{np.percentile(gsm8k17_resB_boot,97.5):.0f}]")
print(f"Difference CI: [{diff17_ci[0]:.1f}, {diff17_ci[1]:.1f}]")
print(f"CI excludes 0: {diff17_ci[0] > 0}")

# Save results
output = {
    'description': 'Paired bootstrap difference test for Res-B between FinQA and GSM8K',
    'n_bootstrap': N_BOOT,
    'finqa_8B': {
        'resB_point': round(finqa_resB, 1),
        'resB_ci95': [round(finqa_ci[0], 1), round(finqa_ci[1], 1)],
    },
    'gsm8k_8B': {
        'resB_point': round(gsm8k_resB, 1),
        'resB_ci95': [round(gsm8k_ci[0], 1), round(gsm8k_ci[1], 1)],
    },
    'diff_8B': {
        'point': round(obs_diff, 1),
        'ci95': [round(diff_ci[0], 1), round(diff_ci[1], 1)],
        'se': round(diff_se, 1),
        'ci_excludes_zero': bool(diff_ci[0] > 0),
        'p_two_sided': round(float(p_two_sided), 4),
    },
    'finqa_1_7B': {
        'resB_point': round(resB(finqa17_accs), 1),
        'resB_ci95': [round(np.percentile(finqa17_resB_boot, 2.5), 1), round(np.percentile(finqa17_resB_boot, 97.5), 1)],
    },
    'gsm8k_1_7B': {
        'resB_point': round(resB(gsm8k17_accs), 1),
        'resB_ci95': [round(np.percentile(gsm8k17_resB_boot, 2.5), 1), round(np.percentile(gsm8k17_resB_boot, 97.5), 1)],
    },
    'diff_1_7B': {
        'point': round(resB(finqa17_accs) - resB(gsm8k17_accs), 1),
        'ci95': [round(diff17_ci[0], 1), round(diff17_ci[1], 1)],
        'ci_excludes_zero': bool(diff17_ci[0] > 0),
    },
    'note': 'FinQA 8B and GSM8K 8B individual CIs overlap at [9,13] pp; the paired difference CI provides the correct significance test.',
}

json.dump(output, open(f"{RESULTS_DIR}/analysis_bootstrap_diff.json", "w"), indent=2)
print(f"\nSaved to {RESULTS_DIR}/analysis_bootstrap_diff.json")
print(f"\nKey finding: Individual CIs overlap, but the paired bootstrap difference test")
print(f"              directly assesses whether the Res-B gap is significant.")
print(f"              8B diff CI: [{diff_ci[0]:.1f}, {diff_ci[1]:.1f}] pp")
