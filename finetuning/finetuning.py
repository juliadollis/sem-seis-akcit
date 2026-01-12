o codigo ta correto e calculando a loss so pra resposta?
import os
import json
import random
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import login
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

hf_user = "juliadollis"
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
dataset_path = "juliadollis/sinth_qwen235b_evasion_justifications_all"
epoca = 3

model_short = model_name.split("/")[-1]
repo_base_name = f"{model_short}_{epoca}ep_assistant_only_v2"
hub_model_id = f"{hf_user}/{repo_base_name}"
output_dir = repo_base_name

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

valid_labels = {"Clear Reply", "Clear Non-Reply", "Ambivalent"}

print("Carregando dataset...")
ds = load_dataset(dataset_path, split="train").shuffle(seed=seed)
ds_splits = ds.train_test_split(test_size=0.1, seed=seed)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

tokenizer.chat_template = """{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n'}}{% if message['role'] == 'assistant' %}{% generation %}{{message['content']}}{% endgeneration %}{% else %}{{message['content']}}{% endif %}{{'<|im_end|>\n'}}{% endfor %}"""

def _safe_str(x):
    return x if isinstance(x, str) else ""

def format_chat(example):
    question = _safe_str(example.get("interview_question")).strip()
    answer = _safe_str(example.get("interview_answer")).strip()
    label = _safe_str(example.get("clarity_label")).strip()
    justification = _safe_str(example.get("justificativa")).strip()

    label = " ".join(label.split())

    if not question or not answer or not justification or label not in valid_labels:
        return None

    user_content = PROMPT.format(
        TAXONOMY=TAXONOMY.strip(),
        INTERVIEW_QUESTION=question,
        INTERVIEW_ANSWER=answer
    ).strip()

    assistant_content = json.dumps(
        {"justification": justification, "label": label},
        ensure_ascii=False
    )

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }

print("Formatando datasets...")
train_ds = ds_splits["train"].map(
    format_chat,
    remove_columns=ds_splits["train"].column_names
).filter(lambda x: x is not None)

test_ds = ds_splits["test"].map(
    format_chat,
    remove_columns=ds_splits["test"].column_names
).filter(lambda x: x is not None)

print(f"Treino: {len(train_ds)} | Teste: {len(test_ds)}")

use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype
)

print("Carregando modelo...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=compute_dtype,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM"
)

wandb_api_key = os.getenv("WANDB_API_KEY")
report_to = "wandb" if wandb_api_key else "none"

if report_to == "wandb":
    wandb.init(project=os.getenv("WANDB_PROJECT", "sft"), name=repo_base_name)

args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=epoca,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=not use_bf16,
    bf16=use_bf16,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    max_length=2048,
    packing=False,
    seed=seed,
    report_to=report_to,
    run_name=repo_base_name,
    push_to_hub=True,
    hub_model_id=hub_model_id,
    hub_strategy="every_save",
    dataset_text_field="messages",
    assistant_only_loss=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    args=args,
    peft_config=peft_config,
    processing_class=tokenizer
)

print("Iniciando treinamento...")
trainer.train()

print("Avaliando...")
metrics = trainer.evaluate()
print(metrics)

print("Enviando para o Hub...")
trainer.push_to_hub()

if report_to == "wandb":
    wandb.finish()
