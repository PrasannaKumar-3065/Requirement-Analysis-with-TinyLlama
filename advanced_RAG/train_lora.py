# train_lora.py (Module 14 – LoRA Fine-Tuning)

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# -----------------------------
# Config
# -----------------------------
BASE_MODEL = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # small model for GTX 1050
TRAIN_PATH = os.getenv("TRAIN_PATH", "train_requirements.jsonl")  # JSONL file
OUTPUT_DIR = os.getenv("LORA_REQUIREMENT_MASTER","./lora-requirement-master")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load Model + Tokenizer
# -----------------------------
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# PEFT (LoRA Config)
# -----------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # works for most LLMs
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading dataset...")
dataset = load_dataset("json", data_files=TRAIN_PATH, split="train")

def format_example(example):
    instruction = example["instruction"]
    context = example["input"]
    answer = example["output"]

    text = f"{instruction}\n\n{context}\n\nAnswer:\n{answer}"
    return {"input_ids": tokenizer(text, truncation=True, padding="max_length", max_length=512)["input_ids"],
            "labels": tokenizer(answer, truncation=True, padding="max_length", max_length=512)["input_ids"]}

tokenized_dataset = dataset.map(format_example)

# -----------------------------
# Training
# -----------------------------
args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=20,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    output_dir=OUTPUT_DIR,
    save_strategy="epoch",
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset
)

trainer.train()

# Save final LoRA adapter
model.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter saved at {OUTPUT_DIR}")
