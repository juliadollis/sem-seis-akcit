import os
import random
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import login
import wandb
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

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

hf_user = "juliadollis"
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

model_name = "meta-llama/Llama-3.2-3B-Instruct"
dataset_path = "ailsntua/QEvasion"
epoca = 5

model_short = model_name.split("/")[-1]
repo_base_name = f"{model_short}_{epoca}ep_ok"
hub_model_id = f"{hf_user}/{repo_base_name}"
output_dir = repo_base_name

print("Carregando dataset...")
ds = load_dataset(dataset_path, split="train").shuffle(seed=seed)
ds_splits = ds.train_test_split(test_size=0.1, seed=seed)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def create_prompt_completion(example):
    part_of_interview = example.get("interview_question") or "No additional context provided."
    question = example.get("question") or ""
    interview_answer = example.get("interview_answer") or ""

    base_prompt = ZERO_SHOT.format(
        TAXONOMY=TAXONOMY,
        PART_OF_INTERVIEW=part_of_interview,
        QUESTION=question,
        INTERVIEW_ANSWER=interview_answer
    )

    user_message = f"{base_prompt}\n{RESPONSE_TAG}"

    messages = [
        {"role": "user", "content": user_message}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    label_text = example.get("clarity_label") or ""
    completion_text = f'{{"label": "{label_text}"}}'

    return {
        "prompt": prompt_text,
        "completion": completion_text
    }

print("Aplicando formatação Prompt-Completion...")
ds_train = ds_splits["train"].map(create_prompt_completion)
ds_test = ds_splits["test"].map(create_prompt_completion)

print(f"Treino: {len(ds_train)} | Teste: {len(ds_test)}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

print("Carregando modelo...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.config.use_cache = False

args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=epoca,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    logging_strategy="steps",
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_length=4096,
    packing=False,
    seed=seed,
    report_to="wandb",
    run_name=repo_base_name,
    push_to_hub=True,
    hub_model_id=hub_model_id,
    hub_strategy="every_save"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    args=args
)

print("Iniciando treinamento...")
trainer.train()

print("Avaliando...")
results = trainer.evaluate()
print(results)

print("Enviando para o Hub...")
trainer.push_to_hub()
wandb.finish()
