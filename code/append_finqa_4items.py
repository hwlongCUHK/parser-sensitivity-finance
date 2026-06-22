#!/usr/bin/env python3
"""
Append 4 new FinQA items to bring sample size from 96 to 100.
Requires GPU — loads models and runs inference on the 4 new items, then applies all 5 parsers.

Usage: python append_finqa_4items.py <model> [gpu_id]
  model: 1.7b | 8b | glm
  gpu_id: 0 (default)

Output: updated p4_finqa_<model>_t16384.json and p4_finqa_<model>_parsers.json (now 100 items)
"""
import json, re, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# === Config ===
MODEL_PATHS = {
    "1.7b": "/data/whp/models/qwen3/Qwen3-1.7B",
    "8b": "/data/houwanlong/models/Qwen/Qwen3-8B",
    "glm": "/data/whp/models/GLM-4.7-Flash",
}
BUDGET = 16384
RESULTS_DIR = "results_icaif"
RELEASE_DIR = "../release/data"

model_name = sys.argv[1]
gpu_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# === Load 4 new items ===
new_items = json.load(open("finqa_4new_items.json"))  # must be in cwd

# === Load model ===
path = MODEL_PATHS[model_name]
dm = {"": f"cuda:{gpu_id}"}
m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map=dm,
    trust_remote_code=True, local_files_only=True)
t = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
if t.pad_token is None:
    t.pad_token = t.eos_token

# === PARSERS (identical to p4_multiparser_eval.py) ===
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
    nums = re.findall(r'-?[\d,]+\.?\d*', resp)
    if nums:
        try: return float(nums[-1].replace(',', ''))
        except: return None
    return None

def extract_finance_aware(resp):
    resp_clean = resp.replace('$', '').replace('%', '')
    resp_clean = re.sub(r'\(([\d.]+)\)', r'-\1', resp_clean)
    for suffix, mult in [('million', 1e6), ('billion', 1e9), ('thousand', 1e3), ('M', 1e6), ('B', 1e9), ('K', 1e3)]:
        resp_clean = resp_clean.replace(suffix, f'*{mult:.0f}')
    boxed = re.findall(r'boxed\{([^}]+)\}', resp_clean)
    if boxed:
        nums = re.findall(r'-?[\d,]+\.?\d*', boxed[-1])
        if nums:
            try: return float(nums[0].replace(',', ''))
            except: pass
    eqn = re.findall(r'=\s*(-?[\d,]+\.?\d*)', resp_clean)
    if eqn:
        try: return float(eqn[-1].replace(',', ''))
        except: pass
    nums = re.findall(r'-?[\d,]+\.?\d*', resp_clean)
    if nums:
        try: return float(nums[-1].replace(',', ''))
        except: pass
    return None

PARSERS = {
    'eq_number': extract_eq_number,
    'boxed': extract_boxed,
    'fixed': extract_fixed,
    'last_number': extract_last_number,
    'finance_aware': extract_finance_aware,
}

def matches(pred, gt_str):
    if pred is None: return False
    try:
        gt_str_clean = gt_str.strip().rstrip('%').replace(',','').replace('$','')
        if gt_str_clean.endswith('%'):
            gt_str_clean = gt_str_clean[:-1]
        gt = float(gt_str_clean)
    except: return False
    return abs(pred - gt) < 1e-4

# === Run inference on 4 new items (EXACT prompt format from p1b_finqa_eval.py) ===
new_results = []
for item in new_items:
    ctx = str(item.get('context', ''))[:800]
    tbl = str(item.get('table', ''))[:800]
    prompt = f"Table:\n{tbl}\n\nContext: {ctx}\n\nQuestion: {item['question']}\n\nAnalyze step by step and give the final numeric answer.\n"
    msgs = [{"role": "user", "content": prompt}]
    txt = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = t(txt, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=BUDGET, temperature=0.0, do_sample=False,
                        pad_token_id=t.eos_token_id)
    resp = t.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    new_results.append({"response": resp, "gt": item['ground_truth'], "question": item['question']})
    print(f"[{item['finqa_idx']}] {item['question'][:80]}... done")

# === Append to existing raw data ===
raw_file = f"{RESULTS_DIR}/p4_finqa_{model_name}_t{BUDGET}.json"
existing_raw = json.load(open(raw_file))
existing_raw.extend(new_results)
json.dump(existing_raw, open(raw_file, "w"), indent=2)
print(f"Raw: {len(existing_raw)} items → {raw_file}")

# === Re-run all 5 parsers on the appended data ===
parser_results = {}
all_predictions = {p: {'predictions': [], 'total': len(existing_raw)} for p in PARSERS}
for pname, pfunc in PARSERS.items():
    correct = 0
    for item in existing_raw:
        pred = pfunc(item['response'])
        is_correct = matches(pred, item['gt'])
        all_predictions[pname]['predictions'].append({
            'pred': pred, 'gt': item['gt'], 'correct': is_correct
        })
        if is_correct:
            correct += 1
    acc = correct / len(existing_raw) * 100
    all_predictions[pname]['correct'] = correct
    all_predictions[pname]['accuracy'] = acc
    print(f"{pname}: {correct}/{len(existing_raw)} ({acc:.1f}%)")

parser_file = f"{RESULTS_DIR}/p4_finqa_{model_name}_parsers.json"
json.dump(all_predictions, open(parser_file, "w"), indent=2)
print(f"Parsers: {len(existing_raw)} items → {parser_file}")

# === Also update release data ===
for fname in [raw_file, parser_file]:
    import shutil
    dest = os.path.join(RELEASE_DIR, os.path.basename(fname))
    shutil.copy(fname, dest)
    print(f"Copied to release: {dest}")

print("\nDone. FinQA sample size is now 100.")
