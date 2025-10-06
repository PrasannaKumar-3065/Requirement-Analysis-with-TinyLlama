# build_self_corrective_dataset.py
import json
import random
from pathlib import Path

INPUT_PATH = "contextual_evaluation_report.json"
OUTPUT_PATH = "self_corrective_dataset.jsonl"

if not Path(INPUT_PATH).exists():
    raise FileNotFoundError(f"{INPUT_PATH} not found. Run eval_lora_contextual.py first.")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    report = json.load(f)

details = report["details"]
records = []

def corrective_feedback(style, generated, reference):
    if style == "narrative":
        return (
            "Narrative answer feels too rigid or list-like. "
            "Rewrite it as a flowing explanation that naturally describes the system in prose form."
        )
    elif style == "analytical":
        return (
            "Analytical answer lacks reasoning depth. Add comparison, cause-effect links, or justification."
        )
    elif style == "brief":
        return (
            "Answer is too long for brief style. Condense to one short, context-faithful sentence."
        )
    return "Adjust tone to better match expected style."

for d in details:
    style = d["expected_style"]
    feedback = corrective_feedback(style, d["generated_output"], d["reference_output"])
    improved_instruction = (
        "You are a Requirement Trainer. Review the following model output and rewrite it "
        f"in a clearer {style} style. Follow the feedback hints below strictly."
    )
    improved_input = (
        f"Context:\n{d['input']}\n\n"
        f"Model Output:\n{d['generated_output']}\n\n"
        f"Feedback:\n{feedback}\n\n"
        "Answer:"
    )
    improved_output = d["reference_output"]

    records.append({
        "instruction": improved_instruction,
        "input": improved_input,
        "output": improved_output
    })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✅ Built {len(records)} self-corrective records and saved to {OUTPUT_PATH}")
