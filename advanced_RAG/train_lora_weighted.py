import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# -----------------------------
# Config
# -----------------------------
BASE_MODEL = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
TRAIN_PATH = os.getenv("TRAIN_PATH", "train_weighted.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./lora-requirement-master-v2")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load Model and Tokenizer
# -----------------------------
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

# LoRA Configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading weighted dataset...")
dataset = load_dataset("json", data_files=TRAIN_PATH, split="train")

def format_example(example):
    text = f"{example['instruction']}\n\n{example['input']}\n\nAnswer:\n{example['output']}"
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    labels = tokenized["input_ids"].copy()
    labels = [-100 if t == tokenizer.pad_token_id else t for t in labels]
    tokenized["labels"] = labels

    # Sample weight (default = 1.0)
    weight = 1.0
    if example.get("weight") == "high":
        weight = 1.5
    elif example.get("weight") == "medium":
        weight = 1.0
    elif example.get("weight") == "low":
        weight = 0.5
    tokenized["sample_weight"] = weight

    return tokenized

tokenized_dataset = dataset.map(format_example)

# -----------------------------
# Custom Trainer with Weighted Loss
# -----------------------------
from torch.nn import CrossEntropyLoss

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        weights = inputs.get("weight", torch.ones(labels.shape[0], device=labels.device))

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = loss.view(labels.size(0), -1).mean(dim=1)
        weighted_loss = (loss * weights).mean()

        return (weighted_loss, outputs) if return_outputs else weighted_loss


# -----------------------------
# Training Arguments
# -----------------------------
args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=20,
    num_train_epochs=2,  # shorter pass
    learning_rate=1e-4,
    fp16=True,
    logging_steps=10,
    output_dir=OUTPUT_DIR,
    save_strategy="epoch",
    save_total_limit=2
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
print(f"✅ Weighted LoRA retraining complete. Saved at {OUTPUT_DIR}")
