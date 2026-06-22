#!/usr/bin/env python3
"""
P5: GSM8K parser sensitivity comparison.
Run the same multi-parser analysis on GSM8K (general math) to compare
extraction failure rates against financial benchmarks.

Uses HuggingFace datasets to load GSM8K.

Usage:
  python p5_gsm8k_comparison.py <model> [n_problems] [budget] [gpu_id]
  Models: 30b, 8b, 1.7b
"""
import json, re, os, sys, random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ========== PARSER DEFINITIONS (same as p4) ==========

def extract_eq_number(resp):
    m = re.search(r'=\s*(-?[\d,]+\.?\d*)', resp)
    if m:
        try: return float(m.group(1).replace(',', ''))
        except: return None
    return None

def extract_boxed(resp):
    matches = re.findall(r'boxed\{([^}]+)\}', resp)
    if matches:
        nums = re.findall(r'-?\d+[\d,]*\.?\d*', matches[-1])
        if nums:
            try: return float(nums[0].replace(',', ''))
            except: return None
    return None

def extract_fixed(resp):
    boxed = extract_boxed(resp)
    if boxed is not None:
        return boxed
    eqn = re.findall(r'=\s*(-?[\d,]+\.?\d*)', resp)
    if eqn:
        try: return float(eqn[-1].replace(',', ''))
        except: return None
    return None

def extract_last_number(resp):
    nums = re.findall(r'(?<![a-zA-Z])(-?[\d,]+\.?\d*)(?![a-zA-Z])', resp)
    nums = [n for n in nums if n and len(n.replace(',','').replace('.','').replace('-','')) > 0]
    if nums:
        try: return float(nums[-1].replace(',', ''))
        except: return None
    return None

def extract_finance_aware(resp):
    """Finance-aware parser (same as p4 but less useful on GSM8K — included for comparison)."""
    boxed = extract_boxed(resp)
    if boxed is not None:
        return boxed
    final_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*\$?\s*(-?[\d,]+\.?\d*)',
        r'(?:therefore|thus|so|hence)\s*,?\s*(?:the\s+)?(?:answer|result|value)\s*(?:is|=|:)\s*\$?\s*(-?[\d,]+\.?\d*)',
    ]
    for pat in final_patterns:
        m = re.search(pat, resp, re.IGNORECASE)
        if m:
            try: return float(m.group(1).replace(',', ''))
            except: pass
    eqn = re.findall(r'=\s*(-?[\d,]+\.?\d*)', resp)
    if eqn:
        try: return float(eqn[-1].replace(',', ''))
        except: pass
    return extract_last_number(resp)

PARSERS = {
    'eq_number': extract_eq_number,
    'boxed': extract_boxed,
    'fixed': extract_fixed,
    'last_number': extract_last_number,
    'finance_aware': extract_finance_aware,
}

def matches(pred, gt, tol=0.01):
    if pred is None: return False
    gn = float(gt)
    if abs(gn) < 1e-8:
        return abs(pred - gn) < 0.01
    return abs(pred - gn) / (abs(gn) + 1e-8) < tol


# ========== GSM8K DATA LOADING ==========

def load_gsm8k(n_problems=100):
    """Load GSM8K test set from local JSON file."""
    if os.path.exists("data/gsm8k_test.json"):
        data = json.load(open("data/gsm8k_test.json"))
        random.shuffle(data)
        return data[:n_problems]
    else:
        print("ERROR: data/gsm8k_test.json not found.")
        print("Prepare it locally and upload to server.")
        sys.exit(1)


# ========== MAIN ==========

def main():
    if len(sys.argv) < 2:
        print("Usage: python p5_gsm8k_comparison.py <model> [n_problems] [budget] [gpu_id]")
        print("Models: 30b, 8b, 1.7b")
        sys.exit(1)

    model_name = sys.argv[1]
    n_problems = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    gpu_id = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    MODEL_PATHS = {
        "30b": "/data/whp/models/qwen3/Qwen3-30B-A3B-Thinking-2507",
        "8b": "/data/houwanlong/models/Qwen/Qwen3-8B",
        "1.7b": "/data/whp/models/qwen3/Qwen3-1.7B",
    }

    path = MODEL_PATHS.get(model_name)
    if not path:
        print(f"Unknown model: {model_name}")
        sys.exit(1)

    # Load GSM8K
    print(f"Loading GSM8K ({n_problems} problems)...")
    data = load_gsm8k(n_problems)
    print(f"Loaded {len(data)} problems.")

    # Load model
    print(f"Loading {model_name} from {path}...")
    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16,
        device_map={"": f"cuda:{gpu_id}"},
        trust_remote_code=True, local_files_only=True
    )
    t = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    print("Loaded.")

    raw_responses = []
    gt_list = []
    all_data = []

    for ex in tqdm(data, desc=f"gsm8k_{model_name}"):
        prompt = f"Question: {ex['question']}\n\nSolve step by step and give the final numeric answer.\n"
        msgs = [{"role": "user", "content": prompt}]
        txt = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = t(txt, return_tensors="pt").to(m.device)

        with torch.no_grad():
            out = m.generate(
                **inp, max_new_tokens=budget,
                temperature=0.0, do_sample=False,
                pad_token_id=t.eos_token_id
            )
        resp = t.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        gt = ex["ground_truth"]

        raw_responses.append(resp)
        gt_list.append(gt)
        all_data.append({
            "response": resp,
            "gt": gt,
            "question": ex["question"],
        })

    # Apply all parsers
    print(f"\n=== Multi-Parser Results: {model_name} on GSM8K (n={len(data)}) ===")
    print(f"{'Parser':<16} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print("-" * 44)

    parser_results = {}
    for pname, pfunc in PARSERS.items():
        correct = 0
        preds = []
        for i, (resp, gt) in enumerate(zip(raw_responses, gt_list)):
            pred = pfunc(resp)
            ok = matches(pred, gt)
            correct += int(ok)
            preds.append({'idx': i, 'pred': pred, 'gt': gt, 'correct': ok})

        acc = correct / max(len(data), 1)
        print(f"{pname:<16} {correct:>8} {len(data):>6} {acc:>10.1%}")
        parser_results[pname] = {
            'correct': correct,
            'total': len(data),
            'accuracy': acc,
            'predictions': preds,
        }

    # Compute parser gap (max - min accuracy)
    accs = [r['accuracy'] for r in parser_results.values()]
    gap = max(accs) - min(accs)
    best_parser = max(parser_results, key=lambda k: parser_results[k]['accuracy'])
    worst_parser = min(parser_results, key=lambda k: parser_results[k]['accuracy'])
    print(f"\nParser gap: {gap:.1%} (best={best_parser} {max(accs):.1%}, worst={worst_parser} {min(accs):.1%})")

    # Save
    os.makedirs("results_icaif", exist_ok=True)
    json.dump(all_data, open(f"results_icaif/p5_gsm8k_{model_name}_t{budget}.json", "w"), indent=2)
    json.dump(parser_results, open(f"results_icaif/p5_gsm8k_{model_name}_parsers.json", "w"), indent=2)
    print(f"Saved to results_icaif/p5_gsm8k_{model_name}_*.json")

    # Summary comparison table
    print(f"\n=== COMPARISON SUMMARY ===")
    print(f"If financial benchmarks show 32-45pp parser gap but GSM8K shows <{gap:.0%},")
    print(f"this confirms finance-specific amplification of extraction sensitivity.")


if __name__ == "__main__":
    main()
