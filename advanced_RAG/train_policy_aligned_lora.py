# train_policy_aligned_lora.py
"""
Train a LoRA adapter using a simple preference (DPO-style) objective
on policy_alignment_dataset.jsonl (records with prompt + chosen / rejected).
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
)

from peft import LoraConfig, get_peft_model

# -----------------------
# Config - tune as needed
# -----------------------
BASE_MODEL = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DATA_PATH = os.getenv("POLICY_ALIGNMENT_DATA", "policy_alignment_dataset.jsonl")
OUTPUT_DIR = os.getenv("LORA_POLICY_OUT", "./lora-policy-aligned")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training hyperparams
EPOCHS = int(os.getenv("EPOCHS", "3"))
PER_DEVICE_BATCH = int(os.getenv("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "8"))
LR = float(os.getenv("LR", "2e-4"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "384"))
MARGIN = float(os.getenv("MARGIN", "1.0"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.0"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "0"))
SEED = int(os.getenv("SEED", "42"))

# LoRA config (conservative defaults)
LORA_R = int(os.getenv("LORA_R", "8"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
TARGET_MODULES = os.getenv("LORA_TARGET_MODULES", "q_proj,v_proj").split(",")

# Safety
torch.manual_seed(SEED)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("Device:", DEVICE)
print("Base model:", BASE_MODEL)
print("Data path:", DATA_PATH)
print("Output dir:", OUTPUT_DIR)

if not Path(DATA_PATH).exists():
    raise FileNotFoundError(f"{DATA_PATH} not found. Build it first (build_policy_alignment_dataset.py)")

# -----------------------
# Dataset
# -----------------------
@dataclass
class PairRecord:
    prompt: str
    chosen: str
    rejected: str

class PolicyPairDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 384):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records: List[PairRecord] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                prompt = j.get("instruction", "").strip() + "\n\n" + j.get("input", "").strip() + "\n\nAnswer:\n"
                chosen = j.get("chosen", "").strip()
                rejected = j.get("rejected", "").strip()
                if prompt and chosen and rejected:
                    self.records.append(PairRecord(prompt=prompt, chosen=chosen, rejected=rejected))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {"prompt": r.prompt, "chosen": r.chosen, "rejected": r.rejected}

def collate_fn(batch, tokenizer, max_length):
    # We'll return tokenized full sequences + masks that mark the "answer" portion.
    # For each item: prompt + chosen, prompt + rejected
    input_prompts = [b["prompt"] for b in batch]
    chosen_texts = [p + b["chosen"] for p, b in zip(input_prompts, batch)]
    rejected_texts = [p + b["rejected"] for p, b in zip(input_prompts, batch)]

    # Tokenize full sequences
    chosen_tok = tokenizer(chosen_texts, truncation=True, padding="longest", max_length=max_length, return_tensors="pt")
    rejected_tok = tokenizer(rejected_texts, truncation=True, padding="longest", max_length=max_length, return_tensors="pt")

    # To compute logprob *only* on the answer part, create labels where prompt tokens are -100
    # Identify prompt token lengths (we re-tokenize prompt alone to get its length)
    prompt_tok = tokenizer(input_prompts, truncation=True, padding="longest", max_length=max_length, return_tensors="pt")
    prompt_lens = (prompt_tok["attention_mask"].sum(dim=1)).tolist()

    # Make labels such that tokens corresponding to prompt are -100
    def make_labels(full_input_ids, prompt_lens):
        labels = full_input_ids.clone()
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100  # mask prompt tokens
        return labels

    chosen_labels = make_labels(chosen_tok["input_ids"], prompt_lens)
    rejected_labels = make_labels(rejected_tok["input_ids"], prompt_lens)

    batch_out = {
        "chosen_input_ids": chosen_tok["input_ids"],
        "chosen_attention_mask": chosen_tok["attention_mask"],
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_tok["input_ids"],
        "rejected_attention_mask": rejected_tok["attention_mask"],
        "rejected_labels": rejected_labels,
    }
    return batch_out

# -----------------------
# Load tokenizer and model + apply LoRA
# -----------------------
print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.float16 if DEVICE == "cuda" and torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype=dtype
)

# Apply LoRA
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# If device_map="auto" placed model on device already; ensure model is on DEVICE
try:
    model.to(DEVICE)
except Exception:
    pass

# -----------------------
# Prepare dataset and dataloader
# -----------------------
print("Loading dataset...")
dataset = PolicyPairDataset(DATA_PATH, tokenizer, max_length=MAX_LENGTH)
print(f"Records: {len(dataset)}")
dataloader = DataLoader(dataset, batch_size=PER_DEVICE_BATCH, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer, MAX_LENGTH), num_workers=0)

# -----------------------
# Optimizer & scheduler
# -----------------------
# Only optimize trainable parameters (LoRA)
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

total_steps = (len(dataloader) * EPOCHS) // max(1, GRAD_ACCUM)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=max(1, total_steps),
)

# -----------------------
# Utility: compute log-prob of answer (sum logp of answer tokens)
# -----------------------
def compute_logprob(input_ids, attention_mask, labels):
    """
    Given a tokenized full sequence with labels that are -100 for prompt tokens
    and real ids for answer tokens, compute the *sum* log-prob of answer tokens.
    We'll use model(..., labels=labels) which returns average loss over *active* tokens,
    so sum_logprob = - loss * num_active_tokens
    """
    # forward
    outputs = model(input_ids=input_ids.to(model.device),
                    attention_mask=attention_mask.to(model.device),
                    labels=labels.to(model.device))
    # outputs.loss is average negative log-likelihood over active tokens
    loss = outputs.loss  # scalar
    # Count active tokens
    active_tokens = (labels != -100).sum().to(loss.device).clamp(min=1)
    sum_nll = loss * active_tokens  # total negative log-likelihood
    logprob = -sum_nll  # sum log-prob
    return logprob, active_tokens.item()

# -----------------------
# Training loop
# -----------------------
print("Starting preference training...")
global_step = 0
model.train()
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    for step, batch in enumerate(dataloader, start=1):
        # compute chosen logprob
        logp_chosen, active_chosen = compute_logprob(
            batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_labels"]
        )
        # compute rejected logprob
        logp_rejected, active_rejected = compute_logprob(
            batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["rejected_labels"]
        )

        # We want to maximize (logp_chosen - logp_rejected).
        # Use margin-based loss: max(0, margin - (logp_chosen - logp_rejected))
        diff = (logp_chosen - logp_rejected)
        loss = F.relu(torch.tensor(MARGIN, device=diff.device) - diff)

        # normalize by number of sequences in batch (because logp is sum over tokens)
        loss = loss / PER_DEVICE_BATCH

        loss.backward()

        if (step % GRAD_ACCUM) == 0 or (step == len(dataloader)):
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / max(1, len(dataloader))
    print(f"Epoch {epoch}/{EPOCHS} — avg_loss: {avg_epoch_loss:.4f} — global_step: {global_step}")

# -----------------------
# Save adapter only
# -----------------------
print("Saving LoRA adapter to", OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
print("Done.")
