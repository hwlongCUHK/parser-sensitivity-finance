#!/usr/bin/env python3
"""
P4: Multi-parser evaluation — apply 5 parsers to EXISTING model outputs.
Adds last-number and finance-aware parsers to the existing =NUMBER, boxed, fixed.
Also applies all parsers to FinQA outputs.

Usage: python p4_multiparser_eval.py
  (no model loading — re-parses saved outputs)
"""
import json, re, os, sys
import numpy as np
from pathlib import Path

# ========== PARSER DEFINITIONS ==========

def extract_eq_number(resp):
    """=NUMBER: first number after equals sign."""
    m = re.search(r'=\s*(-?[\d,]+\.?\d*)', resp)
    if m:
        val = m.group(1).replace(',', '')
        try: return float(val)
        except: return None
    return None

def extract_boxed(resp):
    """\\boxed{}: content inside boxed macro."""
    matches = re.findall(r'boxed\{([^}]+)\}', resp)
    if matches:
        nums = re.findall(r'-?\d+[\d,]*\.?\d*', matches[-1])
        if nums:
            try: return float(nums[0].replace(',', ''))
            except: return None
    return None

def extract_fixed(resp):
    """Fixed: boxed first, then =NUMBER fallback (last match)."""
    boxed = extract_boxed(resp)
    if boxed is not None:
        return boxed
    eqn = re.findall(r'=\s*(-?[\d,]+\.?\d*)', resp)
    if eqn:
        try: return float(eqn[-1].replace(',', ''))
        except: return None
    return None

def extract_last_number(resp):
    """Last-number: extract the last numeric token in the entire response."""
    # Find all standalone numbers (not inside words)
    nums = re.findall(r'(?<![a-zA-Z])(-?[\d,]+\.?\d*)(?![a-zA-Z])', resp)
    # Filter out empty strings and very short matches
    nums = [n for n in nums if n and len(n.replace(',','').replace('.','').replace('-','')) > 0]
    if nums:
        try: return float(nums[-1].replace(',', ''))
        except: return None
    return None

def extract_finance_aware(resp):
    """Finance-aware: handles currency, %, magnitude, parenthetical negatives."""
    # 1. Try boxed first
    boxed = extract_boxed(resp)
    if boxed is not None:
        return boxed

    # 2. Look for "the answer is X" or "final answer: X" patterns
    final_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*\$?\s*(-?[\d,]+\.?\d*)\s*(%|million|billion|M|B|K|thousand)?',
        r'(?:therefore|thus|so|hence)\s*,?\s*(?:the\s+)?(?:answer|result|value)\s*(?:is|=|:)\s*\$?\s*(-?[\d,]+\.?\d*)\s*(%|million|billion|M|B|K|thousand)?',
    ]
    for pat in final_patterns:
        m = re.search(pat, resp, re.IGNORECASE)
        if m:
            val = m.group(1).replace(',', '')
            try:
                v = float(val)
                suffix = m.group(2) if m.group(2) else ''
                if suffix.lower() in ('million', 'm'):
                    v *= 1e6
                elif suffix.lower() in ('billion', 'b'):
                    v *= 1e9
                elif suffix.lower() in ('thousand', 'k'):
                    v *= 1e3
                return v
            except:
                pass

    # 3. Try =NUMBER (last match)
    eqn = re.findall(r'=\s*\$?\s*(-?[\d,]+\.?\d*)\s*(%|million|billion|M|B|K|thousand)?', resp)
    if eqn:
        val, suffix = eqn[-1]
        try:
            v = float(val.replace(',', ''))
            if suffix.lower() in ('million', 'm'):
                v *= 1e6
            elif suffix.lower() in ('billion', 'b'):
                v *= 1e9
            elif suffix.lower() in ('thousand', 'k'):
                v *= 1e3
            return v
        except:
            pass

    # 4. Handle parenthetical negatives: (2.1) -> -2.1
    paren = re.findall(r'\((\d+[\d,]*\.?\d*)\)', resp)
    if paren:
        # Check if it looks like accounting negative (near end of response)
        last_paren = paren[-1]
        try: return -float(last_paren.replace(',', ''))
        except: pass

    # 5. Fallback to last number
    return extract_last_number(resp)

def matches(pred, gt, tol=0.01):
    """Match with relative tolerance."""
    if pred is None: return False
    if isinstance(gt, str):
        nums = re.findall(r'-?\d+\.?\d*', gt)
        if not nums: return False
        gt = float(nums[0])
    gn = float(gt)
    if abs(gn) < 1e-8:
        return abs(pred - gn) < 0.01
    return abs(pred - gn) / (abs(gn) + 1e-8) < tol


# ========== MAIN EVALUATION ==========

PARSERS = {
    'eq_number': extract_eq_number,
    'boxed': extract_boxed,
    'fixed': extract_fixed,
    'last_number': extract_last_number,
    'finance_aware': extract_finance_aware,
}

def load_model_outputs(results_dir, pattern):
    """Load saved model outputs that contain raw responses."""
    files = list(Path(results_dir).glob(pattern))
    return files

def evaluate_with_parsers(data, raw_responses, gt_list, model_name, dataset_name):
    """Apply all parsers to raw responses and compute accuracy."""
    results = {pname: {'correct': 0, 'total': 0, 'predictions': []} for pname in PARSERS}

    for i, (resp, gt) in enumerate(zip(raw_responses, gt_list)):
        for pname, pfunc in PARSERS.items():
            pred = pfunc(resp)
            ok = matches(pred, gt)
            results[pname]['correct'] += int(ok)
            results[pname]['total'] += 1
            results[pname]['predictions'].append({
                'idx': i, 'pred': pred,
                'gt': (lambda s: (float(s) if re.match(r'^-?\d+\.?\d*$', s) else None) if s else None)(re.sub(r'[,%$]', '', str(gt).strip().rstrip('%'))) if gt is not None else None,
                'correct': ok
            })

    print(f"\n=== Multi-Parser Results: {model_name} on {dataset_name} ===")
    print(f"{'Parser':<16} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print("-" * 44)
    for pname in PARSERS:
        r = results[pname]
        acc = r['correct'] / max(r['total'], 1)
        print(f"{pname:<16} {r['correct']:>8} {r['total']:>6} {acc:>10.1%}")

    return results


def run_from_saved_outputs():
    """Re-parse existing saved outputs with all parsers."""
    results_dir = "results_icaif"
    all_results = {}

    # Find all saved result files
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.json'):
            continue

        fpath = os.path.join(results_dir, fname)
        data = json.load(open(fpath))

        # Skip files without raw responses
        if not data or not isinstance(data, list):
            continue
        if 'response' not in data[0] and 'raw_response' not in data[0]:
            continue

        resp_key = 'response' if 'response' in data[0] else 'raw_response'
        gt_key = 'gt' if 'gt' in data[0] else 'ground_truth'

        raw_responses = [d.get(resp_key, '') for d in data]
        gt_list = [d.get(gt_key) for d in data]

        if not any(raw_responses):
            continue

        model_name = fname.replace('.json', '')
        results = evaluate_with_parsers(data, raw_responses, gt_list, model_name, fname)
        all_results[model_name] = results

    # Save combined results
    summary = {}
    for model_name, results in all_results.items():
        summary[model_name] = {
            pname: {
                'correct': r['correct'],
                'total': r['total'],
                'accuracy': r['correct'] / max(r['total'], 1)
            }
            for pname, r in results.items()
        }

    os.makedirs(results_dir, exist_ok=True)
    json.dump(summary, open(f"{results_dir}/p4_multiparser_summary.json", "w"), indent=2)
    print(f"\nSummary saved to {results_dir}/p4_multiparser_summary.json")
    return summary


def run_live_eval(model_name, dataset, budget=16384, gpu_id=0):
    """Run live model inference and apply all parsers."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm

    # Model paths
    MODEL_PATHS = {
        "30b": "/data/whp/models/qwen3/Qwen3-30B-A3B-Thinking-2507",
        "8b": "/data/houwanlong/models/Qwen/Qwen3-8B",
        "1.7b": "/data/whp/models/qwen3/Qwen3-1.7B",
        "glm": "/data/whp/models/GLM-4.7-Flash",
    }

    path = MODEL_PATHS.get(model_name)
    if not path:
        print(f"Unknown model: {model_name}")
        return

    # Load dataset
    if dataset == "finance":
        data = json.load(open("data/hard.json"))
        import random; random.seed(42); random.shuffle(data)
        data = data[:100]
        dataset_name = "FinanceReasoning"
    elif dataset == "finqa":
        data = json.load(open("data/finqa_test.json"))
        import random; random.seed(42); random.shuffle(data)
        data = data[:96]
        dataset_name = "FinQA"
    else:
        print(f"Unknown dataset: {dataset}")
        return

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

    for ex in tqdm(data, desc=f"{model_name}_{dataset}"):
        if dataset == "finqa":
            ctx = str(ex.get('context', ''))[:800]
            tbl = str(ex.get('table', ''))[:800]
            prompt = f"Table:\n{tbl}\n\nContext: {ctx}\n\nQuestion: {ex['question']}\n\nAnalyze step by step and give the final numeric answer.\n"
        else:
            ctx = str(ex.get("context", ""))[:1500]
            prompt = f"Context:\n{ctx}\n\nQuestion: {ex['question']}\n\nAnalyze step by step and give the final numeric answer.\n"

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
            "gt": float(gt) if isinstance(gt, (int, float)) else gt,
            "question": ex.get("question", ""),
        })

    # Apply all parsers
    results = evaluate_with_parsers(data, raw_responses, gt_list, model_name, dataset_name)

    # Save with raw responses for future re-parsing
    os.makedirs("results_icaif", exist_ok=True)
    json.dump(all_data, open(f"results_icaif/p4_{dataset}_{model_name}_t{budget}.json", "w"), indent=2)

    # Save parser results
    parser_results = {}
    for pname, r in results.items():
        parser_results[pname] = {
            'correct': r['correct'],
            'total': r['total'],
            'accuracy': r['correct'] / max(r['total'], 1),
            'predictions': r['predictions'],
        }
    json.dump(parser_results, open(f"results_icaif/p4_{dataset}_{model_name}_parsers.json", "w"), indent=2)
    print(f"Saved to results_icaif/p4_{dataset}_{model_name}_*.json")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python p4_multiparser_eval.py reparse          # Re-parse saved outputs")
        print("  python p4_multiparser_eval.py live <model> <dataset> [budget] [gpu_id]")
        print("  Models: 30b, 8b, 1.7b, glm")
        print("  Datasets: finance, finqa")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "reparse":
        run_from_saved_outputs()
    elif mode == "live":
        model = sys.argv[2]
        dataset = sys.argv[3]
        budget = int(sys.argv[4]) if len(sys.argv) > 4 else 16384
        gpu_id = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        run_live_eval(model, dataset, budget, gpu_id)
    else:
        print(f"Unknown mode: {mode}")
