#!/usr/bin/env python3
"""Counterfactual: strip financial formatting from model outputs, re-run parsers.
   If Res-B shrinks → financial surface forms are the cause.
   If Res-B stays → something else (answer complexity, reasoning difficulty)."""
import json, re, os

RESULTS_DIR = "results_icaif"

# ========== PARSERS (same as experiments) ==========
def extract_eq_number(resp):
    m = re.search(r'=\s*(-?[\d,]+\.?\d*)', resp)
    return float(m.group(1).replace(',','')) if m else None

def extract_boxed(resp):
    m = re.findall(r'boxed\{([^}]+)\}', resp)
    if m:
        n = re.findall(r'-?\d+[\d,]*\.?\d*', m[-1])
        return float(n[0].replace(',','')) if n else None
    return None

def extract_fixed(resp):
    b = extract_boxed(resp)
    if b is not None: return b
    eq = re.findall(r'=\s*(-?[\d,]+\.?\d*)', resp)
    return float(eq[-1].replace(',','')) if eq else None

def extract_last_number(resp):
    nums = re.findall(r'(?<![a-zA-Z])(-?[\d,]+\.?\d*)(?![a-zA-Z])', resp)
    nums = [n.replace(',','') for n in nums if n and n.replace(',','').replace('.','').replace('-','')]
    return float(nums[-1]) if nums else None

def extract_finance_aware(resp):
    b = extract_boxed(resp)
    if b is not None: return b
    # Try answer keyword patterns
    for pat in [r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*\$?\s*(-?[\d,]+\.?\d*)',
                r'(?:therefore|thus|so|hence)\s*,?\s*(?:the\s+)?(?:answer|result)\s*(?:is|=|:)\s*\$?\s*(-?[\d,]+\.?\d*)']:
        m = re.search(pat, resp, re.I)
        if m: return float(m.group(1).replace(',',''))
    # =NUMBER fallback
    eq = re.findall(r'=\s*(-?[\d,]+\.?\d*)', resp)
    if eq: return float(eq[-1].replace(',',''))
    return extract_last_number(resp)

PARSERS = {
    'eq_number': extract_eq_number, 'boxed': extract_boxed,
    'fixed': extract_fixed, 'last_number': extract_last_number,
    'finance_aware': extract_finance_aware,
}

def matches(pred, gt, tol=0.01):
    if pred is None: return False
    try:
        gn = float(str(gt).strip().rstrip('%').replace(',',''))
    except:
        return False
    if abs(gn) < 1e-8: return abs(pred-gn) < 0.01
    return abs(pred-gn)/(abs(gn)+1e-8) < tol

# ========== FINANCIAL FORMAT STRIPPING ==========
def strip_financial_format(text):
    """Replace financial surface forms with plain numeric equivalents."""
    result = text

    # $X.XM → X.X * 1e6
    result = re.sub(r'\$([\d,]+\.?\d*)\s*[Mm](?:illion)?', lambda m: str(float(m.group(1).replace(',','')) * 1e6), result)
    # $X.XB → X.X * 1e9
    result = re.sub(r'\$([\d,]+\.?\d*)\s*[Bb](?:illion)?', lambda m: str(float(m.group(1).replace(',','')) * 1e9), result)
    # $X.XK → X.X * 1e3
    result = re.sub(r'\$([\d,]+\.?\d*)\s*[Kk](?: thousand)?', lambda m: str(float(m.group(1).replace(',','')) * 1e3), result)
    # $X.X (no suffix) → X.X (just strip $)
    result = re.sub(r'\$([\d,]+\.?\d*)', r'\1', result)

    # X.X% → X.X (strip % sign — parsers handle both)
    result = re.sub(r'([\d,]+\.?\d*)\s*%', r'\1', result)

    # Accounting negatives: (X.X) → -X.X
    result = re.sub(r'\(([\d,]+\.?\d*)\)', r'-\1', result)

    # "X.X million" → X.X * 1e6 (without $)
    result = re.sub(r'([\d,]+\.?\d*)\s+million', lambda m: str(float(m.group(1).replace(',','')) * 1e6), result, flags=re.I)
    result = re.sub(r'([\d,]+\.?\d*)\s+billion', lambda m: str(float(m.group(1).replace(',','')) * 1e9), result, flags=re.I)

    # "X.X thousand" → X.X * 1e3
    result = re.sub(r'([\d,]+\.?\d*)\s+thousand', lambda m: str(float(m.group(1).replace(',','')) * 1e3), result, flags=re.I)

    return result

# ========== MAIN ==========
print("Counterfactual Analysis: Financial Format Stripping")
print("="*60)

for bm_label, fname in [('FinQA', 'p4_finqa_8b_t16384.json'), ('FLARE', 'p6_flare_finqa_8b_t16384.json')]:
    raw_data = json.load(open(f"{RESULTS_DIR}/{fname}"))
    n = len(raw_data)

    # Original Res-B
    orig_accs = {p: [] for p in PARSERS}
    strip_accs = {p: [] for p in PARSERS}

    for ex in raw_data:
        resp = str(ex.get('response', ''))
        gt = ex.get('gt', '')
        stripped_resp = strip_financial_format(resp)

        for pname, pfunc in PARSERS.items():
            orig_accs[pname].append(matches(pfunc(resp), gt))
            strip_accs[pname].append(matches(pfunc(stripped_resp), gt))

    def resB(accs_dict):
        fixed = sum(accs_dict['fixed'])/n*100
        last = sum(accs_dict['last_number'])/n*100
        finaw = sum(accs_dict['finance_aware'])/n*100
        return max(fixed,last,finaw) - min(fixed,last,finaw)

    orig_resB = resB(orig_accs)
    strip_resB = resB(strip_accs)

    print(f"\n{bm_label} (n={n}):")
    print(f"{'Parser':<16} {'Original':>10} {'Stripped':>10} {'Δ':>8}")
    print("-"*46)
    for p in PARSERS:
        o = sum(orig_accs[p])/n*100
        s = sum(strip_accs[p])/n*100
        print(f"{p:<16} {o:>10.1f}% {s:>10.1f}% {s-o:>+8.1f}pp")
    print(f"\n  Res-B: {orig_resB:.1f}pp → {strip_resB:.1f}pp (Δ = {strip_resB-orig_resB:+.1f}pp)")

    # How many questions changed?
    changes = 0
    for p in PARSERS:
        for i in range(n):
            if orig_accs[p][i] != strip_accs[p][i]:
                changes += 1
    print(f"  Extraction changed on {changes} / ({len(PARSERS)} parsers × {n} questions) decisions")
