"""
Controlled-Labeled Requirement Trainer Dataset Generator
Module 8, Lesson 3.4 — Final Narrative Dataset Version
"""

import json, random, time
from tqdm import tqdm
from transformers import pipeline
import re

def extract_and_fix_json(raw_text: str):
    """
    Extracts the most likely JSON object from raw model output and fixes common issues.
    """
    # Extract block between first { and last }
    match = re.search(r"\{.*\}", raw_text, re.S)
    if not match:
        return None
    json_str = match.group(0)

    # Heuristic cleanups
    json_str = json_str.replace("\n", " ").replace("\t", " ")
    json_str = re.sub(r"[\r\f]", " ", json_str)
    json_str = re.sub(r",\s*}", "}", json_str)          # trailing comma fix
    json_str = re.sub(r",\s*\]", "]", json_str)
    json_str = re.sub(r"“|”", '"', json_str)            # curly quotes
    json_str = re.sub(r"‘|’", "'", json_str)
    json_str = re.sub(r'([a-zA-Z0-9_])\s*:\s*(")', r'"\1": \2', json_str)  # missing quotes on keys

    try:
        return json.loads(json_str)
    except Exception:
        # Fallback: Try wrapping in structure manually if needed
        if '"instruction"' in json_str and '"output"' in json_str:
            try:
                fixed = json_str.split("}", 1)[0] + "}"
                return json.loads(fixed)
            except Exception:
                return None
        return None
    
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = 0  # GPU index; -1 for CPU
OUTPUT_FILE = "train_requirements_trainer_labeled.jsonl"
NUM_EXAMPLES = 600  # Start smaller; we’ll scale after validation

generator = pipeline(
    "text-generation",
    model=MODEL,
    device=DEVICE,
    torch_dtype="auto"
)

# -----------------------------------
# Context Pools
# -----------------------------------
POSITIVE_CONTEXTS = [
    "The document management module stores, versions, and retrieves technical documents for each department.",
    "The analytics dashboard must display live KPIs from sales and operations in one unified view.",
    "The HR system must allow employees to update personal data and generate payslips monthly.",
    "The fleet tracking software must show real-time vehicle locations and alert for route deviations.",
]

NEGATIVE_CONTEXTS = [
    "The mobile app will enable users to log in and view dashboards.",
    "The platform will manage user permissions and maintain activity logs.",
    "The chatbot will assist users with basic support queries.",
    "The integration service will connect different microservices together.",
]

CONFLICTIVE_CONTEXTS = [
    "The module must generate reports every 5 minutes, but the performance budget only allows 1 report every 15 minutes.",
    "The project documentation mentions both PostgreSQL and MongoDB as the main database.",
    "The API must return data within 200 ms, while a parallel spec allows up to 1 second latency.",
    "The data retention section says 3 years in one paragraph and 5 years in another.",
]

QUESTIONS = [
    "Tell me about this module.",
    "What is the main functionality described?",
    "Are there any conflicts or ambiguities?",
    "What is the performance requirement?",
    "Who uses this system?",
    "What is the project deadline?",
]

# -----------------------------------
# Build prompt dynamically
# -----------------------------------
def build_prompt(context, question, label):
    persona = (
        "You are a senior requirement trainer mentoring a new analyst. "
        "You must always explain in a natural narrative tone — not a short answer, but a detailed explanation. "
        "If something is not specified, explicitly say 'This detail is not specified in the requirement.' "
        "If conflicts exist, describe them clearly and suggest clarifications."
    )

    return f"""
{persona}

Generate exactly one JSON object with these keys:
- "instruction": "You are a Requirement Trainer. Teach the requirement."
- "input": "Context: {context}\\n\\nQuestion: {question}"
- "output": a *narrative explanation* (4–8 sentences) that:
   * describes the requirement and its purpose in plain English,
   * points out conflicts or ambiguities if present,
   * gracefully handles missing information,
   * uses a conversational, trainer-like tone (like explaining to a junior),
   * ends with a confident confirmation of understanding.
- "category": "{label}"

Output only valid JSON.
"""


# -----------------------------------
# Main generation loop
# -----------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i in tqdm(range(NUM_EXAMPLES), desc="Generating trainer-style dataset"):
        r = random.random()
        if r < 0.4:
            ctx = random.choice(POSITIVE_CONTEXTS)
            label = "positive"
        elif r < 0.7:
            ctx = random.choice(NEGATIVE_CONTEXTS)
            label = "negative"
        else:
            ctx = random.choice(CONFLICTIVE_CONTEXTS)
            label = "conflictive"

        q = random.choice(QUESTIONS)
        prompt = build_prompt(ctx, q, label)

        result = generator(
            prompt,
            max_new_tokens=400,
            temperature=0.8,
            top_p=0.9,
            do_sample=True
        )[0]["generated_text"]

        obj = extract_and_fix_json(result)
        print(obj)  # For debugging
        if obj:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()
        else:
            print(f"⚠️ Invalid JSON at sample {i}")
        time.sleep(0.1)

print(f"✅ Controlled-labeled narrative dataset saved to {OUTPUT_FILE}")
