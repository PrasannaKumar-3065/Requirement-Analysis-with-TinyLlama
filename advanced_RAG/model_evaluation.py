import json
import re
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# -----------------------------
# Helper functions
# -----------------------------

def normalize_text(text):
    """
    Lowercase, remove punctuation, extra spaces, and convert numbers to digits if needed.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    # Optional: Convert words to numbers here if needed
    return text

def exact_match(pred, true):
    return normalize_text(pred) == normalize_text(true)

def token_f1(pred, true):
    pred_tokens = normalize_text(pred).split()
    true_tokens = normalize_text(true).split()
    common = Counter(pred_tokens) & Counter(true_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(true_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def semantic_similarity(pred, true, model):
    emb_pred = model.encode([pred])
    emb_true = model.encode([true])
    return cosine_similarity(emb_pred, emb_true)[0][0]

# -----------------------------
# Load data
# -----------------------------

# Example JSONL file: each line is a JSON object with keys: 'predicted', 'expected', 'context'
file_path = "predictions.jsonl"

data = []
with open(file_path, "r") as f:
    for line in f:
        data.append(json.loads(line))

# Load sentence-transformer model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Evaluation
# -----------------------------
results = []

for item in data:
    pred = item.get("predicted", "")
    true = item.get("expected", "")
    
    # Remove sources or extra info if present
    pred_clean = re.split(r"Sources?:", pred)[0].strip()
    
    em = exact_match(pred_clean, true)
    f1 = token_f1(pred_clean, true)
    sem = semantic_similarity(pred_clean, true, model)
    
    results.append({
        "predicted": pred_clean,
        "expected": true,
        "exact_match": em,
        "token_f1": f1,
        "semantic_score": sem
    })

# -----------------------------
# Summary statistics
# -----------------------------
num_samples = len(results)
em_avg = sum(r['exact_match'] for r in results) / num_samples
f1_avg = sum(r['token_f1'] for r in results) / num_samples
sem_avg = sum(r['semantic_score'] for r in results) / num_samples

print(f"Samples evaluated: {num_samples}")
print(f"Exact Match: {em_avg:.2f}")
print(f"Token F1: {f1_avg:.2f}")
print(f"Semantic Similarity: {sem_avg:.2f}")

# Optional: save detailed results
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
