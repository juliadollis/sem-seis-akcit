import os
import json
import random
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

TAXONOMY = """
1. Clear Reply - The information requested is explic-
itly stated (in the requested form)
2. Clear Non-Reply - The information requested is not given
at all due to ignorance, need for clarification or declining to
answer
3. Ambiguous - The information requested is given in
an incomplete way e.g. the answer is too general, partial,
implicit, dodging or deflection.
"""

ZERO_SHOT = """
Based on a segment of the interview in
which the interviewer poses a series of questions, classify the
type of response provided by the interviewee for the following
question using the following taxonomy and then provide a
chain of thought explanation for your decision:
{TAXONOMY}
You are required to respond with a single term corre-
sponding to the Taxonomy code and only.

### Part of the interview ###
{PART_OF_INTERVIEW}

### Question ###
{QUESTION}

### Answer ###
{INTERVIEW_ANSWER}

Output ONLY a single-line JSON exactly matching: {{"label": "<one of the taxonomy labels>"}}
"""

RESPONSE_TAG = "### JSON Output:"

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset_path = "ailsntua/QEvasion"
model_id = "juliadollis/Llama-3.2-3B-Instruct_1ep_ok"
pred_repo_id = f"{model_id}_test_predictions"

device = "cuda" if torch.cuda.is_available() else "cpu"
max_seq_length = 4096
max_new_tokens = 64

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

ds = load_dataset(dataset_path, split="test").shuffle(seed=seed)
ds_test = ds

valid_labels = ["Clear Reply", "Clear Non-Reply", "Ambiguous"]
valid_label_set = set(valid_labels)

def filter_valid(example):
    label = example.get("clarity_label")
    q = example.get("question")
    a = example.get("interview_answer")
    return label in valid_label_set and q is not None and a is not None

ds_test = ds_test.filter(filter_valid)

def build_prompt(example):
    part_of_interview = example.get("interview_question") or "No additional context provided."
    question = example.get("question") or ""
    answer = example.get("interview_answer") or ""
    base_prompt = ZERO_SHOT.format(
        TAXONOMY=TAXONOMY,
        PART_OF_INTERVIEW=part_of_interview,
        QUESTION=question,
        INTERVIEW_ANSWER=answer,
    )
    user_message = f"{base_prompt}\n{RESPONSE_TAG}"
    messages = [
        {"role": "user", "content": user_message}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    example["prompt_infer"] = prompt
    return example

ds_test = ds_test.map(build_prompt)

model = AutoPeftModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
model.eval()

def extract_label(text: str) -> str:
    try:
        first_line = text.split("\n")[0]
        parsed = json.loads(first_line)
        cand = parsed.get("label", "")
        if cand in valid_label_set:
            return cand
    except Exception:
        pass
    normalized = text.lower()
    for c in valid_labels:
        if c.lower() in normalized:
            return c
    return "Ambiguous"

true_labels = []
pred_labels = []

batch_size = 4
prompts = ds_test["prompt_infer"]
clarities = ds_test["clarity_label"]

for i in range(0, len(prompts), batch_size):
    batch_prompts = prompts[i : i + batch_size]
    batch_true = clarities[i : i + batch_size]
    enc = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
    gen_ids = outputs[:, input_ids.shape[1] :]
    decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    for t, d in zip(batch_true, decoded):
        true_labels.append(t)
        pred_labels.append(extract_label(d))

label_to_idx = {l: i for i, l in enumerate(valid_labels)}
n_labels = len(valid_labels)
conf_matrix = np.zeros((n_labels, n_labels), dtype=int)

for t, p in zip(true_labels, pred_labels):
    if t not in label_to_idx or p not in label_to_idx:
        continue
    ti = label_to_idx[t]
    pi = label_to_idx[p]
    conf_matrix[ti, pi] += 1

def compute_f1_macro(conf):
    n = conf.shape[0]
    f1s = []
    for k in range(n):
        tp = conf[k, k]
        fp = conf[:, k].sum() - tp
        fn = conf[k, :].sum() - tp
        precision = 0.0 if tp + fp == 0 else tp / float(tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / float(tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        f1s.append(f1)
    return float(sum(f1s) / len(f1s))

f1_macro = compute_f1_macro(conf_matrix)
correct = 0
total = 0
for t, p in zip(true_labels, pred_labels):
    if t in label_to_idx and p in label_to_idx:
        total += 1
        if t == p:
            correct += 1
accuracy = correct / total if total > 0 else 0.0

print("Labels:", valid_labels)
print("Matriz de confusão:")
print(conf_matrix)
print("Accuracy:", accuracy)
print("F1 macro:", f1_macro)

print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)

report_data = []
for i, label in enumerate(valid_labels):
    tp = conf_matrix[i, i]
    fp = conf_matrix[:, i].sum() - tp
    fn = conf_matrix[i, :].sum() - tp
    tn = conf_matrix.sum() - tp - fp - fn
    support = tp + fn
    precision = 0.0 if tp + fp == 0 else tp / float(tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / float(tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    report_data.append({
        'Class': label,
        'Precision': f'{precision:.4f}',
        'Recall': f'{recall:.4f}',
        'F1-Score': f'{f1:.4f}',
        'Support': support
    })

total_support = sum([r['Support'] for r in report_data])
macro_precision = sum([float(r['Precision']) for r in report_data]) / len(report_data)
macro_recall = sum([float(r['Recall']) for r in report_data]) / len(report_data)
macro_f1 = sum([float(r['F1-Score']) for r in report_data]) / len(report_data)

report_data.append({
    'Class': 'macro avg',
    'Precision': f'{macro_precision:.4f}',
    'Recall': f'{macro_recall:.4f}',
    'F1-Score': f'{macro_f1:.4f}',
    'Support': total_support
})

weighted_precision = sum([float(r['Precision']) * r['Support'] for r in report_data[:-1]]) / total_support
weighted_recall = sum([float(r['Recall']) * r['Support'] for r in report_data[:-1]]) / total_support
weighted_f1 = sum([float(r['F1-Score']) * r['Support'] for r in report_data[:-1]]) / total_support

report_data.append({
    'Class': 'weighted avg',
    'Precision': f'{weighted_precision:.4f}',
    'Recall': f'{weighted_recall:.4f}',
    'F1-Score': f'{weighted_f1:.4f}',
    'Support': total_support
})

print("\n --------------------")
print(model_id)
print("\n --------------------")
print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
print("-"*70)
for row in report_data:
    print(f"{row['Class']:<20} {row['Precision']:>10} {row['Recall']:>10} {row['F1-Score']:>10} {row['Support']:>10}")

print("="*70)
print(f"Accuracy: {accuracy:.4f}")
print("="*70)

df_test = ds_test.to_pandas()
df_test["clarity_pred"] = pred_labels

pred_dataset = Dataset.from_pandas(df_test)
pred_dataset.push_to_hub(pred_repo_id)

print("\nDataset de predições enviado para:", pred_repo_id)
