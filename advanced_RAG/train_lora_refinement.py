# train_lora_refinement.py
# Fine-tune a TinyLlama LoRA adapter on the contrastive dataset generated earlier.
# Saves only the LoRA adapter to OUTPUT_DIR.
#
# Notes:
# - This script expects contrastive_train_dataset.jsonl created by build_contrastive_dataset.py
# - Designed for constrained GPUs: small max_length, per-device batch=1, gradient accumulation.

import os
from pathlib import Path
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# -----------------------------
# Config
# -----------------------------
BASE_MODEL = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DATA_PATH = os.getenv("CONTRASTIVE_PATH", "contrastive_train_dataset.jsonl")
OUTPUT_DIR = os.getenv("LORA_REFINEMENT_OUT", "./lora-requirement-master-v4")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tune these for memory constraints
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "384"))
PER_DEVICE_BATCH = int(os.getenv("PER_DEVICE_BATCH", "1"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "8"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))

# LoRA hyperparams (smaller r for lower memory)
LORA_R = int(os.getenv("LORA_R", "8"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))

print("Device:", DEVICE)
print("Base model:", BASE_MODEL)
print("Data path:", DATA_PATH)
print("Output dir:", OUTPUT_DIR)

if not Path(DATA_PATH).exists():
    raise FileNotFoundError(f"{DATA_PATH} not found. Run build_contrastive_dataset.py first.")

# -----------------------------
# Load tokenizer & base model
# -----------------------------
print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# choose dtype based on availability
dtype = torch.float16 if (DEVICE == "cuda" and torch.cuda.is_available()) else torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",            # let accelerate place layers
    low_cpu_mem_usage=True,
    dtype=dtype
)

# -----------------------------
# Apply LoRA
# -----------------------------
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],  # adapt for model internals if needed
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# try enabling gradient checkpointing to lower mem usage (optional)
# try:
#     model.gradient_checkpointing_enable()
# except Exception:
#     pass
model.config.use_cache = False

# -----------------------------
# Load dataset & preprocess
# -----------------------------
print("Loading contrastive dataset...")
ds = load_dataset("json", data_files=DATA_PATH, split="train")

def make_prompt(example):
    # The contrastive dataset contains instruction/input/previous_output/expected_output/feedback_generated
    # We'll train the model to produce feedback_generated given instruction+input+previous_output+expected_output
    prompt = (
        "You are a Requirement Trainer. Rewrite and improve the model output below.\n\n"
        "Context & Instruction:\n"
        f"{example.get('instruction','You are a Requirement Trainer.')}\n\n"
        f"Input:\n{example.get('input','')}\n\n"
        f"Previous Output:\n{example.get('previous_output','')}\n\n"
        f"Expected Output:\n{example.get('expected_output','')}\n\n"
        "Produce an improved, natural narrative explanation (one paragraph). Answer:\n"
    )
    return prompt

def preprocess(example):
    prompt = make_prompt(example)
    target = example.get("feedback_generated", "").strip()
    # we will train to generate the target, so we append it in the labels
    full = prompt + target
    tokenized = tokenizer(full, truncation=True, padding="max_length", max_length=MAX_LENGTH)
    labels = tokenized["input_ids"].copy()
    # Optionally mask prompt tokens from loss (train only on target). But for causal LM we usually keep the whole sequence.
    # If you want to mask the prompt part, compute prompt_token_len and set labels[:prompt_len] = -100
    # For simplicity, keep full sequence as labels (trainer will handle causal shift)
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

print("Tokenizing dataset...")
tokenized = ds.map(preprocess, remove_columns=ds.column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=None, padding="longest")

# -----------------------------
# TrainingArguments
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
    remove_unused_columns=False,   # keep labels
    dataloader_pin_memory=False,
    report_to="none",
    dataloader_num_workers=0,
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=data_collator
)

# -----------------------------
# Train
# -----------------------------
print("Starting fine-tuning...")
trainer.train()

# -----------------------------
# Save LoRA adapter only
# -----------------------------
print("Saving LoRA adapter to", OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
print("Done.")
