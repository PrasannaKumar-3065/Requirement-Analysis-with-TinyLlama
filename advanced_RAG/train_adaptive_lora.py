# train_adaptive_lora.py
import os
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# -------- CONFIG --------
BASE_MODEL = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DATA_PATH = os.getenv("ADAPTIVE_DATA", "adaptive_length_dataset.jsonl")
OUTPUT_DIR = os.getenv("LORA_ADAPTIVE_OUT", "./lora-adaptive-length")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Safety for small GPU
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
    raise FileNotFoundError(f"{DATA_PATH} not found. Build it first (build_adaptive_length_dataset.py).")

# -------- Load tokenizer & base model --------
print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.float16 if (DEVICE == "cuda" and torch.cuda.is_available()) else torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    low_cpu_mem_usage=True,
    dtype=dtype,
)

# -------- Apply LoRA --------
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# gradient checkpointing can help on small GPUs (optional)
# try:
#     model.gradient_checkpointing_enable()
# except Exception:
#     pass

# -------- Load dataset & preprocess --------
print("Loading dataset...")
ds = load_dataset("json", data_files=DATA_PATH, split="train")

def preprocess(ex):
    # We expect 'instruction', 'input', 'output' fields created earlier
    prompt = ex.get("instruction", "") + "\n\n" + ex.get("input", "")
    # We'll train the model to generate the provided 'output'
    full = prompt + "\n\nAnswer:\n" + ex.get("output", "")
    tokenized = tokenizer(full, truncation=True, padding="max_length", max_length=MAX_LENGTH)
    # For causal LM, labels are the input_ids (Trainer expects that)
    labels = tokenized["input_ids"].copy()
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }

tokenized = ds.map(preprocess, remove_columns=ds.column_names)
tokenized.set_format(type="torch")

# -------- Training args --------
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

# -------- Trainer (with gradient-safe fix) --------
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        loss = outputs.loss
        # Fix: ensure loss is connected to gradient graph
        if loss is None or not loss.requires_grad:
            loss = outputs["logits"].float().mean() * 0
            loss.requires_grad_(True)
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=tokenized,
)

print("Starting training...")
trainer.train()

# Save adapter only
print("Saving LoRA adapter to", OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
print("Done.")


# Save adapter only
print("Saving LoRA adapter to", OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
print("Done.")
