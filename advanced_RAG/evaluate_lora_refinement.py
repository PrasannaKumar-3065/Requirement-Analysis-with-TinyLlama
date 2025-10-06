import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from bert_score import score as bert_score
from rouge import Rouge
from peft import PeftModel

# -----------------------------
# CONFIG
# -----------------------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTERS = {
    "v2": "./lora-requirement-master-v2",
    "v3": "./lora-requirement-master-v3",
    "v4": "./lora-requirement-master-v4"
}
TEST_DATA_PATH = "test_requirements.jsonl"
OUTPUT_PATH = "refinement_evaluation_report.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
rouge = Rouge()

# -----------------------------
# Load test set
# -----------------------------
with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

print(f"Loaded {len(test_data)} test samples.")

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_adapter(name, adapter_path):
    print(f"\n🔹 Evaluating adapter: {adapter_path}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=400,  # ⬆️ allow longer completions
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )

    preds, refs, details = [], [], []

    for sample in test_data:
        prompt = (
            "You are a Requirement Trainer. Analyze the requirement below and answer clearly.\n\n"
            + sample["input"]
            + "\n\nAnswer:\n"
        )
        try:
            output = gen(prompt)[0]["generated_text"].split("Answer:")[-1].strip()
        except Exception:
            output = ""

        # Replace empty predictions with a placeholder
        if not output.strip():
            output = "[NO OUTPUT]"

        preds.append(output)
        refs.append(sample["expected_output"])
        details.append({
            "input": sample["input"],
            "output": output,
            "expected_output": sample["expected_output"]
        })

    # Compute metrics safely
    try:
        rouge_scores = rouge.get_scores(preds, refs, avg=True)
        (P, R, F1) = bert_score(preds, refs, lang="en", rescale_with_baseline=True)
        bert_f1 = float(F1.mean())
    except Exception as e:
        print(f"⚠️ Metric computation failed for {name}: {e}")
        rouge_scores = {"rouge-l": {"f": 0.0}}
        bert_f1 = 0.0

    return {
        "rougeL": rouge_scores["rouge-l"]["f"],
        "bertscore_f1": bert_f1,
        "details": details
    }

# -----------------------------
# Main Loop
# -----------------------------
reports = {}
for name, path in ADAPTERS.items():
    try:
        reports[name] = evaluate_adapter(name, path)
    except Exception as e:
        print(f"⚠️ Skipped {name}: {e}")

# -----------------------------
# Compare Improvements
# -----------------------------
if "v4" in reports and "v3" in reports:
    improvement = {
        "ROUGE-L": reports["v4"]["rougeL"] - reports["v3"]["rougeL"],
        "BERTScore-F1": reports["v4"]["bertscore_f1"] - reports["v3"]["bertscore_f1"]
    }
else:
    improvement = {}

# -----------------------------
# Save Report
# -----------------------------
result = {"reports": reports, "improvement": improvement}
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print("\n📊 Evaluation complete.")
print(json.dumps(improvement, indent=2))
print(f"✅ Report saved to {OUTPUT_PATH}")
