#!/usr/bin/env python3
"""Append 4 new FinQA items to GLM (8192 budget, different from 16384 for Qwen)."""
import json, re, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BUDGET = 8192
GPU_ID = 0  # CUDA_VISIBLE_DEVICES handles physical mapping

new_items = json.load(open("experiments/finqa_4new_items.json"))
existing = json.load(open("results_icaif/p4_finqa_glm_t8192.json"))
print(f"Existing: {len(existing)} items, adding 4 new")

path = "/data/whp/models/GLM-4.7-Flash"
m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map={"":"cuda:0"},
    trust_remote_code=True, local_files_only=True)
t = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
if t.pad_token is None:
    t.pad_token = t.eos_token

for item in new_items:
    ctx = str(item["context"])[:800]
    tbl = str(item["table"])[:800]
    prompt = f"Table:\n{tbl}\n\nContext: {ctx}\n\nQuestion: {item['question']}\n\nAnalyze step by step and give the final numeric answer.\n"
    msgs = [{"role": "user", "content": prompt}]
    txt = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = t(txt, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=BUDGET, temperature=0.0, do_sample=False,
                        pad_token_id=t.eos_token_id)
    resp = t.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    existing.append({"response": resp, "gt": item["ground_truth"], "question": item["question"]})
    print(f"Done: {item['question'][:60]}...")

json.dump(existing, open("results_icaif/p4_finqa_glm_t8192.json", "w"), indent=2)
print(f"Saved raw: {len(existing)} items")

# Parsers
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
    'eq_number': extract_eq_number, 'boxed': extract_boxed,
    'fixed': extract_fixed, 'last_number': extract_last_number,
    'finance_aware': extract_finance_aware,
}

def matches(pred, gt_str):
    if pred is None: return False
    try:
        gt_str_clean = gt_str.strip().rstrip('%').replace(',','').replace('$','')
        if gt_str_clean.endswith('%'): gt_str_clean = gt_str_clean[:-1]
        gt = float(gt_str_clean)
    except: return False
    return abs(pred - gt) < 1e-4

all_preds = {p: {'predictions': [], 'total': len(existing)} for p in PARSERS}
for pname, pfunc in PARSERS.items():
    correct = 0
    for item in existing:
        pred = pfunc(item['response'])
        ok = matches(pred, item['gt'])
        all_preds[pname]['predictions'].append({'pred': pred, 'gt': item['gt'], 'correct': ok})
        if ok: correct += 1
    acc = correct / len(existing) * 100
    all_preds[pname]['correct'], all_preds[pname]['accuracy'] = correct, acc
    print(f"{pname}: {correct}/{len(existing)} ({acc:.1f}%)")

json.dump(all_preds, open("results_icaif/p4_finqa_glm_parsers.json", "w"), indent=2)
print("Done — GLM 100 items saved.")
