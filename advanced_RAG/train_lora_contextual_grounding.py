# train_lora_contextual_grounding.py
"""
Fine-tune a LoRA adapter to improve context-grounded style behavior.

Expect:
 - dataset: contextual_grounding_dataset.jsonl (jsonl with fields: instruction, input, output)
 - base model: set via BASE_MODEL env or default TinyLlama/TinyLlama-1.1B-Chat-v1.0
 - result: adapter saved to OUTPUT_DIR (./lora-requirement-master-v5)
"""

import os
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

# -----------------------------
# Config (tweak if needed)
# -----------------------------
BASE_MODEL = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DATA_PATH = os.getenv("CONTEXTUAL_GROUNDING_DATA", "contextual_grounding_dataset.jsonl")
OUTPUT_DIR = os.getenv("LORA_CONTEXTUAL_OUT", "./lora-requirement-master-v5")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Safe settings for small GPU (8GB)
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
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

if not Path(DATA_PATH).exists():
    raise FileNotFoundError(f"{DATA_PATH} not found. Run the dataset builder first.")

# -----------------------------
# Tokenizer + Base model
# -----------------------------
print("Loading tokenizer and base model (this may take a while)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# prefer dtype argument; use float16 on CUDA if available
dtype = torch.float16 if (DEVICE == "cuda" and torch.cuda.is_available()) else torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    low_cpu_mem_usage=True,
    dtype=dtype,
)

# -----------------------------
# Apply LoRA (small r to fit memory)
# -----------------------------
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],  # adjust if your model differs
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# optional: enable gradient checkpointing if you need extra memory-saving
# try:
#     model.gradient_checkpointing_enable()
# except Exception:
#     pass

# -----------------------------
# Load dataset and preprocess
# -----------------------------
print("Loading dataset...")
ds = load_dataset("json", data_files=DATA_PATH, split="train")

def build_prompt(example):
    # We include instruction so the model knows style condition is present
    # example["input"] already contains context + question + expected style
    instruction = example.get("instruction", "You are a Requirement Trainer. Answer in the requested style and stay grounded in the given context.")
    return instruction + "\n\n" + example["input"] + "\n\nAnswer:\n" + example["output"]

def preprocess(example):
    prompt = build_prompt(example)
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=MAX_LENGTH)
    labels = tokenized["input_ids"].copy()
    # leave labels as is for causal LM training
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

print("Tokenizing dataset...")
tokenized = ds.map(preprocess, remove_columns=ds.column_names)
# keep as default (dict of lists); Trainer will convert to tensors as needed

data_collator = DataCollatorForSeq2Seq(tokenizer, model=None, padding="longest")

# -----------------------------
# Training arguments
# -----------------------------
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
    remove_unused_columns=False,  # causal LM needs labels present
    dataloader_num_workers=0,
    report_to="none",
)

# -----------------------------
# Custom Trainer: compliant compute_loss signature
# -----------------------------
class CustomTrainer(Trainer):
    # signature must be (self, model, inputs, return_outputs=False)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Move tensors to the model device if needed (Trainer normally does this)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

# -----------------------------
# Train
# -----------------------------
print("Starting training...")
trainer.train()

# -----------------------------
# Save LoRA adapter only
# -----------------------------
print("Saving LoRA adapter to", OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
print("Done.")
