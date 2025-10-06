# eval_policy_aligned_lora.py — FINAL PATCHED VERSION
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from evaluate import load

# -----------------------------
# CONFIG
# -----------------------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATHS = {
    "v7_hybrid": "./lora-requirement-master-v7",
    "policy_aligned": "./lora-policy-aligned"
}
TEST_FILE = "test_requirements_ref.jsonl"
OFFLOAD_DIR = "./offload_cache"

# -----------------------------
# LOAD TEST DATA
# -----------------------------
def load_test_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if "input" not in ex:
                continue
            data.append(ex)
    print(f"📄 Loaded {len(data)} test samples.\n")
    return data

samples = load_test_data(TEST_FILE)

# -----------------------------
# LOAD METRICS
# -----------------------------
rouge = load("rouge")
bertscore = load("bertscore")

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def safe_get_reference(ex):
    """Handle both 'output' and 'expected_output' keys."""
    for key in ("expected_output", "output", "reference", "label"):
        if key in ex:
            return ex[key]
    return ""

# -----------------------------
# MODEL LOADING
# -----------------------------
def load_model_with_offload(adapter_path):
    print(f"🔹 Loading adapter from {adapter_path}")
    os.makedirs(OFFLOAD_DIR, exist_ok=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            offload_folder=OFFLOAD_DIR,
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = PeftModel.from_pretrained(model, adapter_path, device_map="auto")
        model.eval()
        print(f"✅ Adapter loaded successfully with hybrid GPU+CPU offloading.\n")
        return model, tokenizer

    except Exception as e:
        print(f"⚠️ Adapter load failed ({e}). Falling back to base model.")
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return model, tokenizer

# -----------------------------
# EVALUATION
# -----------------------------
def evaluate_adapter(name, adapter_path):
    print(f"\n🔹 Evaluating adapter: {name} -> {adapter_path}")
    model, tokenizer = load_model_with_offload(adapter_path)

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    preds, refs = [], []

    for ex in tqdm(samples, desc=f"Evaluating {name}"):
        prompt = f"You are a Requirement Trainer. Analyze:\n\n{ex['input']}\n\nAnswer:\n"
        out = gen(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
        if "Answer:" in out:
            out = out.split("Answer:")[-1].strip()
        preds.append(out)
        refs.append(safe_get_reference(ex))

    # compute metrics only if lengths match and refs not empty
    if len(preds) != len(refs) or not any(refs):
        print("⚠️ Skipping metric computation (invalid refs or mismatch).")
        return {"ROUGE-L": None, "BERTScore-F1": None}

    rouge_res = rouge.compute(predictions=preds, references=refs)
    bert_res = bertscore.compute(predictions=preds, references=refs, lang="en")

    results = {
        "ROUGE-L": rouge_res["rougeL"],
        "BERTScore-F1": sum(bert_res["f1"]) / len(bert_res["f1"])
    }

    print(json.dumps(results, indent=2))
    out_path = f"{name}_policy_eval_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Report saved to {out_path}\n")
    return results

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    for name, path in ADAPTER_PATHS.items():
        evaluate_adapter(name, path)
