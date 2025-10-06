# eval_lora_self_corrective.py
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from bert_score import score as bert_score
from rouge import Rouge
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTERS = {
    "v5_contextual": "./lora-adaptive-length",
    "v6_self_corrective": "./lora-requirement-master-v6"
}
TEST_PATH = "contextual_evaluation_report.json"
OUTPUT_PATH = "self_corrective_evaluation_report.json"

DEVICE = 0 if torch.cuda.is_available() else -1
rouge = Rouge()

# -----------------------------
# Load test data
# -----------------------------
if not Path(TEST_PATH).exists():
    raise FileNotFoundError(f"{TEST_PATH} not found. Run eval_lora_contextual.py first.")

with open(TEST_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)["details"]

print(f"📄 Loaded {len(test_data)} test samples for evaluation.")

# -----------------------------
# Evaluation helper
# -----------------------------
def evaluate_adapter(name, adapter_path):
    print(f"\n🔹 Evaluating adapter: {name} ({adapter_path})")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.load_adapter(adapter_path, adapter_name="eval_adapter")
    model.set_adapter("eval_adapter")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=200,
        temperature=0.3,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    preds, refs = [], []
    details = []

    for idx, sample in enumerate(test_data, start=1):
        prompt = (
            "You are a Requirement Trainer. Answer in the expected style.\n\n"
            f"{sample['input']}\nExpected style: {sample['expected_style']}\n\nAnswer:\n"
        )
        out = gen(prompt, num_return_sequences=1)[0]["generated_text"]
        answer = out.split("Answer:")[-1].strip()

        preds.append(answer)
        refs.append(sample["reference_output"])

        # style scoring — crude lexical heuristic
        expected_style = sample["expected_style"]
        style_score = 0
        if expected_style == "brief":
            style_score = 1.0 if len(answer.split()) <= 15 else 0.5
        elif expected_style == "narrative":
            style_score = 1.0 if any(word in answer.lower() for word in ["the module", "allows", "ensures", "designed"]) else 0.5
        elif expected_style == "analytical":
            style_score = 1.0 if any(word in answer.lower() for word in ["because", "therefore", "conflict", "requirement"]) else 0.5

        details.append({
            "id": idx,
            "input": sample["input"],
            "expected_style": expected_style,
            "generated_output": answer,
            "reference_output": sample["reference_output"],
            "style_score": style_score
        })
        print(f"🧩 {idx}. Style={expected_style} | StyleScore={style_score:.2f}")

    # Compute text similarity metrics
    rouge_scores = rouge.get_scores(preds, refs, avg=True)
    (P, R, F1) = bert_score(preds, refs, lang="en", rescale_with_baseline=True)

    result = {
        "rougeL": rouge_scores["rouge-l"]["f"],
        "bertscore_f1": float(F1.mean()),
        "avg_style_score": sum(d["style_score"] for d in details) / len(details),
        "details": details
    }

    return result


# -----------------------------
# Main loop
# -----------------------------
reports = {}
for name, path in ADAPTERS.items():
    try:
        reports[name] = evaluate_adapter(name, path)
    except Exception as e:
        print(f"⚠️ Skipped {name}: {e}")

# -----------------------------
# Compare v6 vs v5
# -----------------------------
if "v6_self_corrective" in reports and "v5_contextual" in reports:
    improvement = {
        "ROUGE-L": reports["v6_self_corrective"]["rougeL"] - reports["v5_contextual"]["rougeL"],
        "BERTScore-F1": reports["v6_self_corrective"]["bertscore_f1"] - reports["v5_contextual"]["bertscore_f1"],
        "StyleScore": reports["v6_self_corrective"]["avg_style_score"] - reports["v5_contextual"]["avg_style_score"],
    }
else:
    improvement = {}

# -----------------------------
# Save final report
# -----------------------------
final_report = {"reports": reports, "improvement": improvement}
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=2)

print("\n📊 Evaluation complete.")
print(json.dumps(improvement, indent=2))
print(f"✅ Report saved to {OUTPUT_PATH}")
