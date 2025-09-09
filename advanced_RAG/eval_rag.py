# eval_rag.py (Module 9 - Hallucination Reduction & Faithfulness Evaluation)

import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
import os
import re
from difflib import SequenceMatcher

load_dotenv()  # Loads variables from .env into environment

BASE_MODEL = os.getenv("BASE_MODEL")
ADAPTER_PATH = os.getenv("ADAPTER_PATH")
EVAL_DATA_PATH = os.getenv("EVAL_DATA_PATH")

# -----------------------------
# Load Model + Tokenizer
# -----------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
# (LoRA weights merged already during training in previous module)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

# -----------------------------
# Helper Functions
# -----------------------------

STRICT_PROMPT = """You are a strict assistant.
You will ONLY answer using the provided context below.
If the answer is not present, say "Not found in context."
Give the answer in this format:

Answer: <short answer>
Sources: <doc_ids or Not found>

Context:
{context}

Question: {question}
"""

def generate_answer(question, context_chunks):
    """Generate grounded answer from context."""
    context_text = "\n".join(context_chunks)
    prompt = STRICT_PROMPT.format(question=question, context=context_text)

    outputs = pipe(
        prompt,
        max_new_tokens=200,
        do_sample=False,
        temperature=0.0,
    )
    text = outputs[0]["generated_text"]

    # Extract only "Answer:" part for evaluation
    answer = "Not found"
    for line in text.splitlines():
        if line.startswith("Answer:"):
            answer = line.replace("Answer:", "").strip()
            break
    return answer, text


def exact_match(pred, gold):
    return pred.strip().lower() == gold.strip().lower()


def f1_score(pred, gold):
    """Simple token overlap F1."""
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


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def token_overlap(a: str, b: str) -> float:
    """Compute token overlap similarity between two strings."""
    a_tokens = set(normalize(a).split())
    b_tokens = set(normalize(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

def is_faithful(answer: str, context_chunks: list[str], threshold: float = 0.6) -> bool:
    """
    Check if the answer is supported by any context chunk.
    Uses fuzzy token overlap + sequence similarity.
    """
    ans_norm = normalize(answer)

    for chunk in context_chunks:
        chunk_norm = normalize(chunk)

        # direct substring check
        if ans_norm in chunk_norm:
            return True

        # fuzzy token overlap
        if token_overlap(ans_norm, chunk_norm) >= threshold:
            return True

        # fallback: sequence similarity
        if SequenceMatcher(None, ans_norm, chunk_norm).ratio() >= threshold:
            return True

    return False


# -----------------------------
# Load Evaluation Data
# -----------------------------
with open(EVAL_DATA_PATH, "r") as f:
    eval_data = json.load(f)

# -----------------------------
# Run Evaluation
# -----------------------------
exact_count = 0
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

    pred, raw_output = generate_answer(question, context_chunks)
    print(f"Predicted: {pred}")
    print(f"Expected : {expected}")

    # Metrics
    em = exact_match(pred, expected)
    f1 = f1_score(pred, expected)
    faithful = is_faithful(pred, context_chunks)

    if em:
        print("✅ Exact match")
        exact_count += 1
    else:
        print("❌ Not exact")

    print(f"F1 score : {f1:.2f}")
    f1_total += f1

    if faithful:
        print("🟢 Faithful")
        faithful_count += 1
    else:
        print("🔴 Hallucination risk")

    print("Raw Output:\n", raw_output)

# -----------------------------
# Final Results
# -----------------------------
print("\n====================")
print(f"Final Results on {n} Qs")
print(f"Exact Match : {exact_count}/{n} ({100*exact_count/n:.2f}%)")
print(f"Avg F1      : {f1_total/n:.2f}")
print(f"Faithful    : {faithful_count}/{n} ({100*faithful_count/n:.2f}%)")
