# eval_lora_hybrid_v7.py
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from bert_score import score as bert_score
from rouge import Rouge

# -----------------------------
# CONFIG
# -----------------------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./lora-requirement-master-v7"
TEST_DATA_PATH = "test_requirements.jsonl"
OUTPUT_PATH = "hybrid_v7_evaluation_report.json"
DEVICE = 0 if torch.cuda.is_available() else -1

rouge = Rouge()

# -----------------------------
# Load test samples
# -----------------------------
with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]
print(f"📄 Loaded {len(test_data)} test samples.")

# -----------------------------
# Load model + adapter
# -----------------------------
print("🔹 Loading hybrid v7 model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, 
    device_map="auto", 
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Try loading adapter if not already merged
try:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("✅ Adapter loaded successfully.")
except Exception:
    print("⚠️ Adapter already merged into base model — continuing.")

# -----------------------------
# Evaluation pipeline
# -----------------------------
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
for i, sample in enumerate(test_data[:10]):  # limit to 10 for quick test
    prompt = (
        "You are a Requirement Trainer. Analyze the requirement below and answer clearly.\n\n"
        f"{sample['input']}\n\nAnswer:\n"
    )
    output = gen(prompt)[0]["generated_text"].split("Answer:")[-1].strip()
    preds.append(output)
    refs.append(sample["expected_output"])
    print(f"\n🧩 {i+1}. Generated:\n{output}\n{'-'*70}")

# -----------------------------
# Compute metrics
# -----------------------------
rouge_scores = rouge.get_scores(preds, refs, avg=True)
(P, R, F1) = bert_score(preds, refs, lang="en", rescale_with_baseline=True)

report = {
    "rougeL": rouge_scores["rouge-l"]["f"],
    "bertscore_f1": float(F1.mean())
}

print("\n📊 Evaluation complete.")
print(json.dumps(report, indent=2))

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(f"✅ Report saved to {OUTPUT_PATH}")
