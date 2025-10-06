# eval_lora_contextual.py
"""
Evaluates the context-grounded LoRA model (v5)
Checks both:
 - contextual accuracy (using ROUGE + BERTScore)
 - style compliance (brief / analytical / narrative)
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from rouge import Rouge
from bert_score import score as bert_score

# -----------------------------
# CONFIG
# -----------------------------
BASE_MODEL = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ADAPTER_PATH = os.getenv("LORA_CONTEXTUAL_OUT", "./lora-requirement-master-v5")
TEST_PATH = os.getenv("CONTEXTUAL_TEST", "contextual_grounding_test.jsonl")
OUTPUT_PATH = os.getenv("CONTEXTUAL_EVAL_OUT", "contextual_evaluation_report.json")

DEVICE = 0 if torch.cuda.is_available() else -1
rouge = Rouge()

# -----------------------------
# Load test set
# -----------------------------
if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"{TEST_PATH} not found. Please provide test samples.")

with open(TEST_PATH, "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

print(f"📄 Loaded {len(test_data)} test samples.")

# -----------------------------
# Load model + adapter
# -----------------------------
print("🔹 Loading model + adapter...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Try to load adapter
try:
    model.load_adapter(ADAPTER_PATH)
    model.set_adapter("default")
    print("✅ Adapter loaded successfully.")
except Exception as e:
    print(f"⚠️ Could not load adapter — continuing with base model: {e}")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=200,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

# -----------------------------
# Evaluation loop
# -----------------------------
preds, refs, details = [], [], []

def evaluate_style(generated: str, expected_style: str) -> float:
    """
    Quick heuristic style check — assigns a compliance score [0-1]
    """
    g = generated.lower()
    s = expected_style.lower()

    if s == "brief":
        # short = <50 words ideally
        return 1.0 if len(g.split()) < 60 else 0.5
    elif s == "analytical":
        # look for explanation structure
        return 1.0 if any(w in g for w in ["therefore", "because", "hence", "analysis", "reason"]) else 0.5
    elif s == "narrative":
        # look for flowing description
        return 1.0 if any(w in g for w in ["story", "describes", "explains", "overall", "illustrates"]) or len(g.split()) > 60 else 0.5
    else:
        return 0.5

print("\n🔍 Running evaluation...\n")
for i, sample in enumerate(test_data, 1):
    prompt = (
        "You are a Requirement Trainer. Analyze the following in the specified style.\n\n"
        + sample["input"]
        + f"\nExpected length: {sample.get('style', 'analytical')}\n\nAnswer:\n"
    )
    output = generator(prompt)[0]["generated_text"].split("Answer:")[-1].strip()

    preds.append(output)
    refs.append(sample["expected_output"])

    style_score = evaluate_style(output, sample.get("style", "analytical"))

    details.append({
        "id": i,
        "input": sample["input"],
        "expected_style": sample.get("style", "analytical"),
        "generated_output": output,
        "reference_output": sample["expected_output"],
        "style_score": style_score
    })

    print(f"🧩 {i}. Style={sample.get('style','analytical')} | StyleScore={style_score:.2f}")
    print("Generated:", output[:180].replace("\n", " ") + ("..." if len(output) > 180 else ""))
    print("-" * 70)

# -----------------------------
# Compute metrics
# -----------------------------
rouge_scores = rouge.get_scores(preds, refs, avg=True)
P, R, F1 = bert_score(preds, refs, lang="en", rescale_with_baseline=True)
avg_style = sum(d["style_score"] for d in details) / len(details)

summary = {
    "rougeL": rouge_scores["rouge-l"]["f"],
    "bertscore_f1": float(F1.mean()),
    "avg_style_score": avg_style,
}

print("\n📊 Evaluation complete.")
print(json.dumps(summary, indent=2))

# -----------------------------
# Save results
# -----------------------------
result = {
    "summary": summary,
    "details": details
}
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(f"✅ Report saved to {OUTPUT_PATH}")
