import json
import re
from tqdm import tqdm
from textblob import TextBlob

# -----------------------------
# Configuration
# -----------------------------
INPUT_FILE = "eval_lora_results.jsonl"   # or .jsonl
OUTPUT_FILE = "refinement_ready.jsonl"

# -----------------------------
# Helper Functions
# -----------------------------
def detect_headings(text):
    return bool(re.search(r"(#+\s|\*|\- )", text))

def detect_conflict_terms(text):
    conflict_terms = ["conflict", "contradict", "inconsistent", "ambiguity", "unclear"]
    return any(term in text.lower() for term in conflict_terms)

def narrative_fluency(text):
    blob = TextBlob(text)
    avg_len = sum(len(s.words) for s in blob.sentences) / max(1, len(blob.sentences))
    sentiment = abs(blob.sentiment.polarity)
    if avg_len < 6 or avg_len > 35:
        return 0.5
    return round(min(1.0, 0.7 + sentiment / 2), 2)

def classify_quality(output_text):
    headings = detect_headings(output_text)
    fluency = narrative_fluency(output_text)
    has_conflict = detect_conflict_terms(output_text)

    if headings:
        label = "improvable"
    elif fluency < 0.6:
        label = "bad"
    else:
        label = "good"

    return {
        "fluency": fluency,
        "has_conflict_terms": has_conflict,
        "label": label
    }

# -----------------------------
# JSON/JSONL Loader (Fix)
# -----------------------------
def load_data_safely(path):
    """Load either a .json or .jsonl file automatically."""
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]

# -----------------------------
# Main Pipeline
# -----------------------------
def build_feedback_pipeline():
    data = load_data_safely(INPUT_FILE)

    refined_samples = []
    for sample in tqdm(data, desc="Evaluating Samples"):
        model_output = sample.get("model_output", "")
        meta = classify_quality(model_output)

        refined_samples.append({
            "instruction": "You are a Requirement Trainer. Teach the requirement in natural narrative style.",
            "input": sample.get("input", ""),
            "output": model_output,
            "quality": meta["label"],
            "fluency": meta["fluency"],
            "has_conflict_terms": meta["has_conflict_terms"]
        })

    filtered = [s for s in refined_samples if s["quality"] in ["good", "improvable"]]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in filtered:
            f.write(json.dumps(entry) + "\n")

    print(f"\n✅ Feedback pipeline completed. Saved {len(filtered)} refinement-ready samples → {OUTPUT_FILE}")

if __name__ == "__main__":
    build_feedback_pipeline()
