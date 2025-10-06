# train_lora_self_eval.py
import os
import json
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# -----------------------------
# Config
# -----------------------------
BASE_MODEL = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
SELF_EVAL_PATH = os.getenv("SELF_EVAL_PATH", "self_eval_dataset.jsonl")
OUTPUT_DIR = os.getenv("LORA_SELF_EVAL_OUT", "./lora-requirement-master-v3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Safety: extremely small batch/seq lengths for 8GB GPUs
MAX_LENGTH = 384
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8  # accumulate to emulate slightly bigger batch
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
LORA_R = 8        # smaller r to reduce memory
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

print("Using device:", DEVICE)
print("Base model:", BASE_MODEL)
print("Self-eval dataset:", SELF_EVAL_PATH)

if not Path(SELF_EVAL_PATH).exists():
    raise FileNotFoundError(f"{SELF_EVAL_PATH} not found. Build it first (build_self_eval_dataset.py).")

# -----------------------------
# Load tokenizer & model
# -----------------------------
print("Loading tokenizer and base model... (this may take a while)")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Use dtype argument instead of deprecated torch_dtype
dtype = torch.float16 if (DEVICE == "cuda" and torch.cuda.is_available()) else torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    low_cpu_mem_usage=True,
    dtype=dtype
)

# -----------------------------
# Apply LoRA
# -----------------------------
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],  # adapt if different model internals
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Verify trainable layers
for n, p in model.named_parameters():
    if p.requires_grad and "lora" in n:
        print(f"✅ Trainable: {n} | shape={tuple(p.shape)}")

# Enable gradient checkpointing (reduces memory, at cost of compute)
# try:
#     model.gradient_checkpointing_enable()
# except Exception:
#     pass

model.config.use_cache = False

# -----------------------------
# Load dataset & preprocess
# -----------------------------
print("Loading dataset...")
ds = load_dataset("json", data_files=SELF_EVAL_PATH, split="train")

def build_input(example):
    # instruction + input + model output already in example["input"]
    prompt = (
        "You are a Requirement Trainer. Self-evaluate the model output below. "
        "Return a short label and a hint in one paragraph. Do NOT use headings.\n\n"
    )
    full = prompt + example["input"] + "\n\nExpected Style: Label: <good|improvable|hallucinated>. Hint: <one-sentence suggestion>\n\nAnswer:\n"
    return full

def preprocess(ex):
    text = build_input(ex)
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH)
    labels = tokenized["input_ids"].copy()
    # We train the model to generate the label+hint; we keep the whole sequence as labels
    # Optionally you might want to shift labels, but Trainer handles causal LM.
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

tokenized = ds.map(preprocess, remove_columns=ds.column_names)

# data collator that does not alter labels (causal LM)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=None, padding="longest")

# -----------------------------
# Training Arguments
# -----------------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=(dtype==torch.float16),
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    report_to="none",
    # reduce CPU memory pressure on small machines:
    dataloader_num_workers=0,
)

# -----------------------------
# Trainer
# -----------------------------
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss



trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=data_collator
)

# -----------------------------
# Train
# -----------------------------
print("Starting training...")
trainer.train()

# -----------------------------
# Save adapter only (keeps base model untouched)
# -----------------------------
print("Saving LoRA adapter to", OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
print("Done.")
