# build_contrastive_dataset.py
# -------------------------------------------------------
# Generates a contrastive refinement dataset using
# feedback from self-eval LoRA (v3) to improve v2 outputs.
# -------------------------------------------------------

import os
import json
from tqdm import tqdm
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# -----------------------------
# Config
# -----------------------------
SELF_EVAL_ADAPTER = "./lora-requirement-master-v3"
SOURCE_REPORT = "comparison_report.json"
OUT_PATH = "contrastive_train_dataset.jsonl"

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Validation
# -----------------------------
if not Path(SOURCE_REPORT).exists():
    raise FileNotFoundError(f"{SOURCE_REPORT} not found. Run compare_evaluation_improvement.py first.")
print(f"✅ Found source report: {SOURCE_REPORT}")

# -----------------------------
# Load model + adapter
# -----------------------------
print("Loading self-eval LoRA (v3)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.float16 if DEVICE == "cuda" else torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    low_cpu_mem_usage=True,
    dtype=dtype
)

# Attach LoRA adapter
model = PeftModel.from_pretrained(base_model, SELF_EVAL_ADAPTER)
model.eval()

# ✅ FIXED: Removed explicit device arg — let Accelerate handle placement
gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.3,
    top_p=0.95,
)

# -----------------------------
# Load source report
# -----------------------------
with open(SOURCE_REPORT, "r", encoding="utf-8") as f:
    report = json.load(f)

base_details = report["reports"]["baseline"]["details"]
weighted_details = report["reports"]["weighted"]["details"]

# -----------------------------
# Build contrastive pairs
# -----------------------------
contrastive_data = []

print("Generating contrastive feedback dataset...")
for base, weighted in tqdm(zip(base_details, weighted_details), total=len(base_details)):
    context = (
        f"Context: {base['input']}\n\n"
        f"Model Output (v2): {base['output']}\n\n"
        f"Expected Output: {base['expected_output']}\n\n"
        f"Evaluate and rewrite the Model Output using the Expected Output as guidance. "
        f"Return an improved, narrative-style version — clear, fluent, and realistic.\n\nAnswer:\n"
    )

    try:
        response = gen(context, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)[0]["generated_text"]
    except Exception as e:
        print(f"⚠️ Generation failed: {e}")
        continue

    contrastive_data.append({
        "instruction": "You are a Requirement Trainer. Rewrite and improve the model output based on feedback.",
        "input": base["input"],
        "previous_output": base["output"],
        "expected_output": base["expected_output"],
        "feedback_generated": response.strip(),
    })

# -----------------------------
# Save result
# -----------------------------
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for item in contrastive_data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"\n✅ Contrastive dataset built: {OUT_PATH}")
print(f"📊 Samples generated: {len(contrastive_data)}")
print("Next step: run `python train_lora_refinement.py` to fine-tune on this dataset.")
