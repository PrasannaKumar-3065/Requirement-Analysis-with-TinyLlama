# eval_rag.py (Module 13 - Requirement Master Trainer Edition, Fixed)

import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os
import re
import torch

load_dotenv()

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = os.getenv("BASE_MODEL")
EVAL_DATA_PATH = os.getenv("EVAL_DATA_PATH")

# -----------------------------
# Load Model + Tokenizer
# -----------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype="auto",
    device_map=DEVICE
)

# Strict Requirement Master prompt
TRAINER_PROMPT = """You are a Requirement Master. 
Your task is to analyze the CONTEXT and QUESTION below.
DO NOT write Q/A pairs. 
DO NOT echo the question or the context. 
Write ONLY in the following sections:

### Summary:
(one-sentence plain English summary of the requirement)

### Requirement:
(clear description of the requirement as stated in context)

### Rationale:
(why this requirement exists; what purpose it serves)

### Conflicts / Ambiguities:
(list any conflicting or ambiguous points in the context; if none, write "None found")

If the requirement is not present in the context, put "Not found" under Requirement and explain in Rationale.

Context:
{context}

Question: {question}
"""

# -----------------------------
# Helper Functions
# -----------------------------
def parse_sections(raw_output: str) -> dict:
    """
    Extract structured sections (Summary, Requirement, Rationale, Conflicts)
    from the model output.
    """
    sections = {"Summary": "", "Requirement": "", "Rationale": "", "Conflicts": ""}

    for key in sections.keys():
        match = re.search(rf"### {key}:\s*(.*?)(?=\n###|\Z)", raw_output, re.S | re.I)
        if match:
            sections[key] = match.group(1).strip()

    # Fill "Not found" if empty
    for k, v in sections.items():
        if not v:
            sections[k] = "Not found"

    return sections


def generate_answer(question, context_chunks):
    """Generate structured requirement explanation from context."""
    # context must not contain Q/A — just factual notes
    context_text = "\n".join(context_chunks)
    prompt = TRAINER_PROMPT.format(question=question, context=context_text)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=True,          # encourage structured fill
        temperature=0.3,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sections = parse_sections(text)

    return sections, text


def f1_score(pred, gold):
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = len(pred_tokens & gold_tokens)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


# -----------------------------
# Load Evaluation Data
# -----------------------------
with open(EVAL_DATA_PATH, "r") as f:
    eval_data = json.load(f)

LOG_PATH = "eval_log_module13.jsonl"
with open(LOG_PATH, "w", encoding="utf-8") as _:
    pass  # reset log file

# -----------------------------
# Run Evaluation
# -----------------------------
f1_total = 0
faithful_count = 0
n = len(eval_data)

for ex in eval_data:
    question = ex["question"]
    expected = ex["answer"]
    context_chunks = ex.get("context", [])

    print("\n====================")
    print(f"Q: {question}")
    print("--------------------")

    sections, raw_output = generate_answer(question, context_chunks)

    # Print structured output
    print("Summary   :", sections["Summary"])
    print("Requirement:", sections["Requirement"])
    print("Rationale :", sections["Rationale"])
    print("Conflicts :", sections["Conflicts"])
    print("Expected  :", expected)

    # Simple F1 on Requirement text vs expected
    f1 = f1_score(sections["Requirement"], expected)
    f1_total += f1

    faithful = sections["Requirement"] != "Not found"
    if faithful:
        faithful_count += 1

    print(f"F1 score  : {f1:.2f}")
    print("Faithful  :", "🟢" if faithful else "🔴")
    print("Raw Output:\n", raw_output)

    # Log JSONL
    log_entry = {
        "question": question,
        "expected": expected,
        "predicted_summary": sections["Summary"],
        "predicted_requirement": sections["Requirement"],
        "predicted_rationale": sections["Rationale"],
        "predicted_conflicts": sections["Conflicts"],
        "f1": round(f1, 2),
        "faithful": faithful,
        "raw_output": raw_output
    }
    with open(LOG_PATH, "a", encoding="utf-8") as logf:
        logf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

# -----------------------------
# Final Results
# -----------------------------
print("\n====================")
print(f"Final Results on {n} Qs")
print(f"Avg F1      : {f1_total/n:.2f}")
print(f"Faithful    : {faithful_count}/{n} ({100*faithful_count/n:.2f}%)")
print(f"\nJSONL logs saved to {LOG_PATH}")
