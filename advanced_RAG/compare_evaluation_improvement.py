import os, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import evaluate

# -------------------------
# Config
# -------------------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTERS = {
    "baseline": "./lora-requirement-master",
    "weighted": "./lora-requirement-master-v2"
}
TEST_FILE = "test_requirements.jsonl"
REPORT_FILE = "comparison_report.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 256

# -------------------------
# Metrics
# -------------------------
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

def generate_output(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model(adapter_path, test_data):
    print(f"\n🔹 Evaluating adapter: {adapter_path}")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    results = []
    preds, refs = [], []

    for item in test_data:
        prompt = f"You are a Requirement Trainer. Explain the requirement.\n\n{item['input']}"
        output = generate_output(model, tokenizer, prompt)
        preds.append(output)
        refs.append(item["expected_output"])
        results.append({"input": item["input"], "output": output, "expected_output": item["expected_output"]})

    rouge_scores = rouge.compute(predictions=preds, references=refs)
    bert_scores = bertscore.compute(predictions=preds, references=refs, lang="en")

    avg_rougeL = rouge_scores["rougeL"]
    avg_bertF1 = sum(bert_scores["f1"]) / len(bert_scores["f1"])
    return {"rougeL": avg_rougeL, "bertscore_f1": avg_bertF1, "details": results}

def main():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    reports = {}
    for name, adapter_path in ADAPTERS.items():
        reports[name] = evaluate_model(adapter_path, test_data)

    improvement = {
        "ROUGE-L": reports["weighted"]["rougeL"] - reports["baseline"]["rougeL"],
        "BERTScore-F1": reports["weighted"]["bertscore_f1"] - reports["baseline"]["bertscore_f1"]
    }

    print("\n📊 Improvement Summary:")
    print(json.dumps(improvement, indent=2))
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump({"reports": reports, "improvement": improvement}, f, indent=2)
    print(f"\n✅ Comparison report saved to {REPORT_FILE}")

if __name__ == "__main__":
    main()
