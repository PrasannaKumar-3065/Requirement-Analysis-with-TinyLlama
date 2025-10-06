import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from evaluate import load
import numpy as np
import json

# -----------------------------
# Config
# -----------------------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_ADAPTER = "./lora-requirement-master"
TEST_PATH = "validation_requirements.jsonl"  # use a small test JSONL
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_FILE = "eval_lora_results.jsonl"
# -----------------------------
# Load Model
# -----------------------------
print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Load Dataset
# -----------------------------
dataset = load_dataset("json", data_files=TEST_PATH, split="train")

# -----------------------------
# Metrics
# -----------------------------
bertscore = load("bertscore")
rouge = load("rouge")

# -----------------------------
# Evaluation Function
# -----------------------------
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=250, temperature=0.6, top_p=0.9)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_sample(example):
    instruction = (
        "You are a Requirement Trainer. "
        "Explain the requirement in a clear, narrative, paragraph style — "
        "do NOT use headings, bullet points, or section titles. "
        "Speak naturally as if mentoring a new analyst."
    )
    prompt = f"{instruction}\n\n{example['input']}\n\nAnswer:"
    prediction = generate_response(prompt)

    refs = [example["output"]]
    preds = [prediction]

    rouge_score = rouge.compute(predictions=preds, references=refs)
    bert_score = bertscore.compute(predictions=preds, references=refs, lang="en")

    return {
        "prediction": prediction,
        "rougeL": rouge_score["rougeL"],
        "bertscore_f1": np.mean(bert_score["f1"])
    }

# -----------------------------
# Run Evaluation
# -----------------------------
results = [evaluate_sample(e) for e in dataset.select(range(5))]

print("\n📊 Evaluation Results:")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i, r in enumerate(results):
        print(f"\nSample {i+1}:")
        print(f"ROUGE-L: {r['rougeL']:.3f} | BERTScore-F1: {r['bertscore_f1']:.3f}")
        print(f"Model Output:\n{r['prediction'][:500]}...")
        obj = {
            "input": dataset[i]["input"],
            "model_output": r["prediction"],
            "expected_output": dataset[i]["output"],
            "rougeL": r["rougeL"],
            "bertscore_f1": r["bertscore_f1"],
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()