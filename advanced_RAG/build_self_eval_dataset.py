import json
from pathlib import Path
from tqdm import tqdm

INPUT_FILE = "comparison_report.json"
OUTPUT_FILE = "self_eval_dataset.jsonl"

if not Path(INPUT_FILE).exists():
    raise FileNotFoundError(f"{INPUT_FILE} not found. Run compare_evaluation_improvement.py first.")

print(f"Loading comparison report: {INPUT_FILE}")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

baseline_samples = data["reports"]["baseline"]["details"]
weighted_samples = data["reports"]["weighted"]["details"]

print("Building self-evaluation dataset...")

def assess_output(model_output, expected_output):
    """
    Very lightweight quality classifier:
    Returns a self-evaluation label and a correction hint.
    """
    model_output = model_output.lower()
    expected_output = expected_output.lower()

    if "the specific information is not" in model_output:
        return "improvable", "Add elaboration or rationale instead of just stating missing info."
    elif len(model_output.split()) < 50:
        return "improvable", "Expand with more narrative flow and contextual explanation."
    elif "###" in model_output or "-" in model_output:
        return "improvable", "Remove heading markers; use natural paragraph form."
    elif "question:" in model_output:
        return "hallucinated", "Avoid asking further questions; focus on answering the given prompt."
    else:
        return "good", "Maintain current fluency and narrative tone."

records = []

for base, weighted in tqdm(zip(baseline_samples, weighted_samples), total=len(baseline_samples)):
    input_text = weighted["input"]
    output_text = weighted["output"]
    expected_text = weighted["expected_output"]

    label, hint = assess_output(output_text, expected_text)

    records.append({
        "instruction": "Self-evaluate this AI-generated requirement analysis.",
        "input": input_text + "\n\nModel Output:\n" + output_text,
        "output": f"Label: {label}\nHint: {hint}",
        "label": label,
        "hint": hint
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ Self-evaluation dataset built: {OUTPUT_FILE}")
print(f"📊 Total Records: {len(records)}")
