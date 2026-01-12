import os
import json
import random
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

dataset_path = "ailsntua/QEvasion"
model_id = "juliadollis/Qwen3-4B-Instruct-2507_10ep_json_just_labelv2_2"
pred_repo_id = f"{model_id}_QEvasion_predictions"
log_path = "relatorio_inferencia_teste.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
max_seq_length = 4096
max_new_tokens = 256
batch_size = 4
debug_n = 5

valid_labels = ["Clear Reply", "Clear Non-Reply", "Ambivalent"]
label_to_idx = {l: i for i, l in enumerate(valid_labels)}
label_aliases = {
    "clear reply": "Clear Reply",
    "clear non-reply": "Clear Non-Reply",
    "ambivalent": "Ambivalent",
    "ambiguous": "Ambivalent"
}

TAXONOMY = """
1. Clear Reply: The answer directly addresses the core intent of the question, providing the specific information or position requested.
2. Clear Non-Reply: The respondent explicitly fails to provide the info (e.g., "I don't know", "I can't say", "I'm not aware").
3. Ambivalent: The respondent evades the core question, using vague, indirect, or pivoting language.
"""

PROMPT = """
You are an expert linguistic annotator. Your task is to classify interview answers based ONLY on the three categories provided below.

### TAXONOMY
{TAXONOMY}

### RULES
- You MUST choose exactly one label.
- Write 2 to 5 sentences explaining your decision based on the core intent of the question.
- Use only information from the question and answer.
- Do not add external facts.
- Output must be a single valid JSON object.
- Do not include any text outside the JSON.

---
INTERVIEW QUESTION:
"{INTERVIEW_QUESTION}"

INTERVIEW ANSWER:
"{INTERVIEW_ANSWER}"
---

### OUTPUT FORMAT
{{
  "justification": "...",
  "label": "Clear Reply" or "Clear Non-Reply" or "Ambivalent"
}}
"""

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
model.config.use_cache = True

ds = load_dataset(dataset_path, split="test").shuffle(seed=seed)

def build_prompt(example):
    q = example.get("interview_question") or ""
    a = example.get("interview_answer") or ""
    text = PROMPT.format(
        TAXONOMY=TAXONOMY.strip(),
        INTERVIEW_QUESTION=q.strip(),
        INTERVIEW_ANSWER=a.strip()
    )
    messages = [{"role": "user", "content": text}]
    example["prompt"] = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    return example

ds = ds.map(build_prompt)

def normalize_label(x):
    if not isinstance(x, str):
        return None
    return label_aliases.get(x.strip().lower())

def extract_first_json(text):
    depth = 0
    start = None
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i+1])
                    if "justification" in obj and "label" in obj:
                        return obj
                except Exception:
                    pass
                start = None
    return None

true_labels = []
pred_labels = []
justifications = []
indices = []

prompts = ds["prompt"]
true_col = "clarity_label_norm" if "clarity_label_norm" in ds.column_names else "clarity_label"
true_vals = ds[true_col]

with open(log_path, "w", encoding="utf-8") as log:
    printed = 0

    for i in tqdm(range(0, len(prompts), batch_size), desc="Batches"):
        batch_prompts = prompts[i:i+batch_size]
        batch_true = true_vals[i:i+batch_size]
        batch_idx = list(range(i, min(i+batch_size, len(prompts))))

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length
        ).to(device)

        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        prompt_lens = enc["attention_mask"].sum(dim=1)

        for j in range(len(batch_prompts)):
            gen = out[j, prompt_lens[j]:]
            decoded = tokenizer.decode(gen, skip_special_tokens=True).strip()

            obj = extract_first_json(decoded)
            pred = normalize_label(obj.get("label")) if obj else None
            just = obj.get("justification") if obj else None
            true = normalize_label(batch_true[j])

            true_labels.append(true)
            pred_labels.append(pred)
            justifications.append(just)
            indices.append(batch_idx[j])

            log.write("\n" + "=" * 120 + "\n")
            log.write("PROMPT:\n")
            log.write(batch_prompts[j] + "\n")
            log.write("\nMODEL OUTPUT:\n")
            log.write(decoded + "\n")
            log.write("\nJSON EXTRAIDO:\n")
            log.write(json.dumps(obj, ensure_ascii=False) + "\n")
            log.write(f"\nLABEL PREDITO: {pred}\n")
            log.write(f"LABEL VERDADEIRO: {true}\n")
            log.write("=" * 120 + "\n")

filtered_true = []
filtered_pred = []

for t, p in zip(true_labels, pred_labels):
    if t in label_to_idx and p in label_to_idx:
        filtered_true.append(t)
        filtered_pred.append(p)

y_true = [label_to_idx[x] for x in filtered_true]
y_pred = [label_to_idx[x] for x in filtered_pred]

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
acc = accuracy_score(y_true, y_pred)
f1m = f1_score(y_true, y_pred, average="macro")
report = classification_report(y_true, y_pred, target_names=valid_labels)

with open(log_path, "a", encoding="utf-8") as log:
    log.write("\n\n==================== METRICS ====================\n")
    log.write("Labels: " + str(valid_labels) + "\n\n")
    log.write("Confusion Matrix:\n")
    log.write(np.array2string(cm) + "\n\n")
    log.write(f"Accuracy: {acc:.4f}\n")
    log.write(f"F1 Macro: {f1m:.4f}\n\n")
    log.write("Classification Report:\n")
    log.write(report + "\n")
    log.write("================================================\n")

print("Accuracy:", acc)
print("F1 Macro:", f1m)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

df = pd.DataFrame({
    "index": indices,
    "label_true": true_labels,
    "label_pred": pred_labels,
    "justification_pred": justifications
})

Dataset.from_pandas(df).push_to_hub(pred_repo_id)

print("Inferência concluída")
print("Relatório salvo em:", log_path)
print("Dataset enviado para:", pred_repo_id)
