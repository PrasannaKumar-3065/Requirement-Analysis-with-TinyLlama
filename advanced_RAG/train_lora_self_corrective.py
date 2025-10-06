# train_lora_self_corrective.py
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

BASE_MODEL = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DATA_PATH = os.getenv("SELF_CORRECTIVE_DATA", "self_corrective_dataset.jsonl")
OUTPUT_DIR = os.getenv("LORA_SELF_CORRECTIVE_OUT", "./lora-requirement-master-v6")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LENGTH = 384
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

print("Device:", DEVICE)
print("Base model:", BASE_MODEL)
print("Training data:", DATA_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.float16 if (DEVICE == "cuda") else torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    low_cpu_mem_usage=True,
    dtype=dtype,
)

# Load LoRA v5 (contextual)
base_model.load_adapter("./lora-adaptive-length", adapter_name="contextual_v5")
base_model.set_adapter("contextual_v5")

# Apply a new refinement LoRA head (v6)
lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()

# Load dataset
ds = load_dataset("json", data_files=DATA_PATH, split="train")

def preprocess(ex):
    full_text = ex["instruction"] + "\n\n" + ex["input"] + "\n" + ex["output"]
    tokens = tokenizer(full_text, truncation=True, padding="max_length", max_length=MAX_LENGTH)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = ds.map(preprocess, remove_columns=ds.column_names)
tokenized.set_format(type="torch")

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=(dtype == torch.float16),
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=tokenized)
print("Starting training...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
print(f"✅ Self-corrective LoRA saved to {OUTPUT_DIR}")
