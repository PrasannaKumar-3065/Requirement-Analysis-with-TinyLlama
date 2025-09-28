# eval_rag.py (Module 11 - Improved evaluation: robust extraction + semantic faithfulness)
import json
import os
import re
from difflib import SequenceMatcher
from typing import List, Tuple, Dict, Any

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import torch

load_dotenv()  # Loads BASE_MODEL, ADAPTER_PATH, EVAL_DATA_PATH from .env

BASE_MODEL = os.getenv("BASE_MODEL")
ADAPTER_PATH = os.getenv("ADAPTER_PATH")  # kept for compatibility (not used if LoRA merged)
EVAL_DATA_PATH = os.getenv("EVAL_DATA_PATH")
RESULTS_OUT = os.getenv("EVAL_RESULTS_OUT", "eval_results_module11.jsonl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype="auto",
    device_map="auto" if DEVICE == "cuda" else None,
)

# If LoRA adapter path is provided and you didn't merge, you could load it here:
# from peft import PeftModel
# model = PeftModel.from_pretrained(model, ADAPTER_PATH)

gen_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Sentence embedding model for semantic similarity
print("Loading sentence-transformer for semantic checks...")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

# Strict prompt you had
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

# -----------------------------
# Text normalization / metrics
# -----------------------------
def normalize(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    # keep punctuation for embeddings, but for token metrics we remove
    return text

def token_f1(pred: str, gold: str) -> float:
    """Simple token-overlap F1 (word-level)."""
    p_tokens = [t for t in re.findall(r"\w+", pred.lower())]
    g_tokens = [t for t in re.findall(r"\w+", gold.lower())]
    if not p_tokens or not g_tokens:
        return 0.0
    common = {}
    for t in p_tokens:
        common[t] = common.get(t, 0) + 1
    match = 0
    for t in g_tokens:
        if common.get(t, 0) > 0:
            match += 1
            common[t] -= 1
    precision = match / len(p_tokens)
    recall = match / len(g_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def exact_match(pred: str, gold: str) -> bool:
    return re.sub(r"\s+", " ", pred.strip().lower()) == re.sub(r"\s+", " ", gold.strip().lower())

# -----------------------------
# Answer extraction utilities
# -----------------------------
ANSWER_LINE_RE = re.compile(r"^\s*Answer\s*:\s*(.+)", flags=re.IGNORECASE)
SOURCES_LINE_RE = re.compile(r"^\s*Sources\s*:\s*(.+)", flags=re.IGNORECASE)

def extract_candidates_from_generated(full_text: str, question: str) -> List[str]:
    """
    Robustly extract candidate answers from generated text.
    Strategy:
      - Find the first "Answer:" occurrences and capture following lines until "Sources:" or blank line cluster.
      - Also capture any other "Answer:" occurrences.
      - If none found, try to take the text after the question token and heuristically pick the first sentence(s).
    """
    candidates = []

    lines = full_text.splitlines()
    # Try to find explicit "Answer:" blocks
    i = 0
    while i < len(lines):
        m = ANSWER_LINE_RE.match(lines[i])
        if m:
            # gather lines from this line onward until Sources: or two consecutive blank lines or next "Answer:"
            ans_lines = [m.group(1).strip()]
            i += 1
            while i < len(lines):
                if SOURCES_LINE_RE.match(lines[i]) or ANSWER_LINE_RE.match(lines[i]):
                    break
                # stop if line is obviously part of the original prompt (e.g., "Context:" or "Question:")
                if lines[i].strip().startswith("Context:") or lines[i].strip().startswith("Question:"):
                    break
                ans_lines.append(lines[i].strip())
                i += 1
            cand = " ".join([l for l in ans_lines if l])
            if cand:
                candidates.append(cand.strip())
            continue
        i += 1

    if candidates:
        # dedupe and return
        seen = set()
        uniq = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    # fallback heuristics: find the last occurrence of the question and take subsequent text
    qi = full_text.rfind(question)
    if qi != -1:
        after = full_text[qi + len(question) :].strip()
    else:
        after = full_text.strip()

    # remove repeated prompt prefix if present
    # trim partial "Sources:" etc.
    after = re.split(r"\bSources\s*:", after, flags=re.IGNORECASE)[0].strip()

    # Take first 1-2 sentences as candidate
    sents = re.split(r'(?<=[.!?])\s+', after)
    if sents:
        cand = " ".join(sents[:2]).strip()
        if cand:
            return [cand]

    # ultimate fallback: return the whole after text trimmed
    if after:
        return [after]

    return ["Not found"]

# -----------------------------
# Faithfulness checker (tightened)
# -----------------------------
def is_faithful(answer: str, context_chunks: List[str], sem_threshold: float = 0.72, token_threshold: float = 0.45, seq_threshold: float = 0.6) -> Tuple[bool, float]:
    """
    Returns (is_faithful, best_semantic_score)
    Logic:
      - Compute semantic similarity between answer and each context chunk using sentence-transformer cosine.
      - If max cosine >= sem_threshold -> faithful.
      - Else fallback: token-overlap F1 >= token_threshold -> faithful.
      - Else fallback: SequenceMatcher ratio >= seq_threshold -> faithful.
    """
    a = normalize(answer)
    if not a or a.lower() in ("not found", "not found in context", ""):
        return False, 0.0

    # semantic
    try:
        emb_a = embedder.encode(a, convert_to_tensor=True)
        chunk_embs = embedder.encode(context_chunks, convert_to_tensor=True)
        cos_scores = util.cos_sim(emb_a, chunk_embs)[0]  # vector of scores
        max_sim = float(cos_scores.max().cpu().numpy())
    except Exception:
        max_sim = 0.0

    if max_sim >= sem_threshold:
        return True, max_sim

    # token overlap fallback (F1 style)
    for chunk in context_chunks:
        f1 = token_f1(answer, chunk)
        if f1 >= token_threshold:
            return True, max_sim

    # sequence similarity fallback
    for chunk in context_chunks:
        if SequenceMatcher(None, answer.lower(), chunk.lower()).ratio() >= seq_threshold:
            return True, max_sim

    return False, max_sim

# -----------------------------
# Generation wrapper
# -----------------------------
def generate_answer(question: str, context_chunks: List[str], max_new_tokens: int = 120) -> Tuple[List[str], str]:
    """
    Returns (candidate_answers, full_generated_text)
    Uses STRICT_PROMPT but robustly extracts candidates.
    """
    context_text = "\n".join(context_chunks)
    prompt = STRICT_PROMPT.format(question=question, context=context_text)

    out = gen_pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,   # deterministic
        # remove temperature/top_p because pipeline may warn they are ignored in some versions
        return_full_text=True,
    )
    full_text = out[0]["generated_text"]
    candidates = extract_candidates_from_generated(full_text, question)
    # ensure uniqueness and trim
    uniq = []
    for c in candidates:
        cleaned = re.sub(r"\s+", " ", c).strip()
        if cleaned and cleaned not in uniq:
            uniq.append(cleaned)
    return uniq, full_text

# -----------------------------
# Load evaluation data
# -----------------------------
with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# -----------------------------
# Run evaluation
# -----------------------------
results = []
exact_count = 0
f1_total = 0.0
faithful_count = 0
n = len(eval_data)

print("\nStarting evaluation ...")
for ex in eval_data:
    question = ex["question"]
    expected = ex["answer"]
    context_chunks = ex.get("context", [])

    print("\n====================")
    print(f"Q: {question}")
    print("--------------------")

    candidates, raw_output = generate_answer(question, context_chunks)
    # Choose best candidate by semantic score against context (higher is better)
    best = None
    best_sem = -1.0
    best_f1 = 0.0
    best_em = False
    best_faith = False

    for cand in candidates:
        faithful, sem_score = is_faithful(cand, context_chunks)
        f1 = token_f1(cand, expected)
        em = exact_match(cand, expected)
        # ranking: prefer faithful with highest semantic; else token-F1
        score = (1.0 if faithful else 0.0) * 10.0 + sem_score + f1
        if best is None or score > ( (1.0 if best_faith else 0.0) * 10.0 + best_sem + best_f1):
            best = cand
            best_sem = sem_score
            best_f1 = f1
            best_em = em
            best_faith = faithful

    if best is None:
        best = "Not found"
        best_sem = 0.0
        best_f1 = 0.0
        best_em = False
        best_faith = False

    print(f"Predicted (best): {best}")
    print(f"Expected       : {expected}")
    print(f"Candidates     : {candidates}")
    print(f"Semantic score : {best_sem:.3f}")
    print(f"Token-F1 vs gold: {best_f1:.3f}")
    print(f"Exact match    : {best_em}")
    print(f"Faithful       : {best_faith}")
    print("Raw Output:\n", raw_output[:4000])  # print truncated raw text for debugging

    # Metrics accumulation
    if best_em:
        exact_count += 1
    f1_total += best_f1
    if best_faith:
        faithful_count += 1

    # Save per-example result
    results.append({
        "question": question,
        "expected": expected,
        "predicted": best,
        "candidates": candidates,
        "semantic_score": best_sem,
        "token_f1": best_f1,
        "exact_match": bool(best_em),
        "faithful": bool(best_faith),
        "raw_output": raw_output,
        "context_used_count": len(context_chunks),
    })

# -----------------------------
# Final Results
# -----------------------------
print("\n====================")
print(f"Final Results on {n} Qs")
print(f"Exact Match : {exact_count}/{n} ({100*exact_count/n:.2f}%)")
print(f"Avg Token-F1: {f1_total/n:.3f}")
print(f"Faithful    : {faithful_count}/{n} ({100*faithful_count/n:.2f}%)")

# Optionally write results to JSONL
with open(RESULTS_OUT, "w", encoding="utf-8") as out_f:
    for r in results:
        out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Saved per-example results to {RESULTS_OUT}")
