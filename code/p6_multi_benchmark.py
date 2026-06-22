#!/usr/bin/env python3
"""
P6: Multi-benchmark parser sensitivity comparison.
Run 5 parsers on SVAMP, ASDiv, MultiArith to compare with financial benchmarks.
Usage: python p6_multi_benchmark.py <model> <dataset> [budget] [gpu_id]
  Models: 1.7b, 8b
  Datasets: svamp, asdiv, multiarith
"""
import json, re, os, sys, random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ========== PARSERS (same as p4/p5) ==========
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
    if boxed is not None: return boxed
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
    boxed = extract_boxed(resp)
    if boxed is not None: return boxed
    for pat in [r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*\$?\s*(-?[\d,]+\.?\d*)',
                r'(?:therefore|thus|so|hence)\s*,?\s*(?:the\s+)?(?:answer|result)\s*(?:is|=|:)\s*\$?\s*(-?[\d,]+\.?\d*)']:
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
    'eq_number': extract_eq_number, 'boxed': extract_boxed,
    'fixed': extract_fixed, 'last_number': extract_last_number,
    'finance_aware': extract_finance_aware,
}

def matches(pred, gt, tol=0.01):
    if pred is None: return False
    gn = float(gt)
    if abs(gn) < 1e-8: return abs(pred - gn) < 0.01
    return abs(pred - gn) / (abs(gn) + 1e-8) < tol

# ========== MAIN ==========
model_name = sys.argv[1]
dataset_name = sys.argv[2]
budget = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
gpu_id = int(sys.argv[4]) if len(sys.argv) > 4 else 0

MODEL_PATHS = {
    "1.7b": "/data/whp/models/qwen3/Qwen3-1.7B",
    "8b": "/data/houwanlong/models/Qwen/Qwen3-8B",
    "glm": "/data/whp/models/GLM-4.7-Flash",
    "30b": "/data/whp/models/qwen3/Qwen3-30B-A3B-Thinking-2507",
}
path = MODEL_PATHS[model_name]
data = json.load(open(f"data/{dataset_name}_test.json"))
random.shuffle(data)
data = data[:100]
print(f"{dataset_name}: {model_name}, {len(data)} problems, budget={budget}")

dm = "auto" if model_name == "30b" else {"": f"cuda:{gpu_id}"}
m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map=dm, trust_remote_code=True, local_files_only=True)
t = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
if t.pad_token is None: t.pad_token = t.eos_token
print("Loaded.")

raw_responses, gt_list, all_data = [], [], []
for ex in tqdm(data, desc=f"{dataset_name}_{model_name}"):
    ctx = ex.get('context', '')
    if ctx:
        prompt = f"Context: {ctx[:800]}\n\nQuestion: {ex['question']}\n\nAnalyze step by step and give the final numeric answer.\n"
    else:
        prompt = f"Question: {ex['question']}\n\nSolve step by step and give the final numeric answer.\n"
    msgs = [{"role": "user", "content": prompt}]
    txt = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = t(txt, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=budget, temperature=0.0, do_sample=False, pad_token_id=t.eos_token_id)
    resp = t.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    raw_responses.append(resp)
    gt_list.append(ex["ground_truth"])
    all_data.append({"response": resp, "gt": ex["ground_truth"], "question": ex["question"]})

# Apply all parsers
print(f"\n=== Multi-Parser Results: {model_name} on {dataset_name} (n={len(data)}) ===")
print(f"{'Parser':<16} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
print("-" * 44)
parser_results = {}
for pname, pfunc in PARSERS.items():
    correct = sum(1 for resp, gt in zip(raw_responses, gt_list) if matches(pfunc(resp), gt))
    acc = correct / len(data)
    print(f"{pname:<16} {correct:>8} {len(data):>6} {acc:>10.1%}")
    parser_results[pname] = {'correct': correct, 'total': len(data), 'accuracy': acc}

accs = [r['accuracy'] for r in parser_results.values()]
print(f"\nParser gap: {max(accs)-min(accs):.1%} (best={max(parser_results, key=lambda k: parser_results[k]['accuracy'])} {max(accs):.1%}, worst={min(parser_results, key=lambda k: parser_results[k]['accuracy'])} {min(accs):.1%})")

os.makedirs("results_icaif", exist_ok=True)
json.dump(all_data, open(f"results_icaif/p6_{dataset_name}_{model_name}_t{budget}.json", "w"), indent=2)
json.dump(parser_results, open(f"results_icaif/p6_{dataset_name}_{model_name}_parsers.json", "w"), indent=2)
print(f"Saved to results_icaif/p6_{dataset_name}_{model_name}_*.json")
