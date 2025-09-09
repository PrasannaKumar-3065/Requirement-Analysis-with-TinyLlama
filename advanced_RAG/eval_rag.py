#!/usr/bin/env python3
# eval_rag.py -- Module 8 evaluation harness (multi-metric, stricter hallucination + rubric)

import os
import re
import json
from collections import Counter
from typing import List, Tuple

from advanced_rag import (
    load_and_chunk_docs,
    build_index,
    retrive,
    rerank,
    extractive_bullet,
    dedup_bullets_semantic,
    advanced_rag,
)

# -------------------------
# Evaluation set
# -------------------------
EVAL_QUESTIONS = [
    {"question": "What is the project deadline?", "expected": "October 2025"},
    {"question": "Who is the client?", "expected": "ACME Corp"},
    {"question": "What are the performance requirements?", "expected": "handle 500 concurrent users; respond within 2 seconds"},
]

if os.path.exists("/home/yesituser/yesitprojects/venv/isoTracker_requirement_AI/advanced_RAG/eval_data.json"):
    try:
        with open("/home/yesituser/yesitprojects/venv/isoTracker_requirement_AI/advanced_RAG/eval_data.json", "r", encoding="utf-8") as f:
            EVAL_QUESTIONS = json.load(f)
    except Exception:
        print("Warning: failed to read eval_data.json, using builtin eval set.")


# -------------------------
# Text utils
# -------------------------
STOPWORDS = set("""
the a an in on at of for with and or to from by as is was are were be been it this that
""".split())

def normalize_text(s: str) -> str:
    s = s or ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tokenize(s: str) -> List[str]:
    return normalize_text(s).split()


def compute_f1(pred_tokens: List[str], gold_tokens: List[str]) -> Tuple[float, float, float]:
    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    pred_cnt = Counter(pred_tokens)
    gold_cnt = Counter(gold_tokens)
    common = sum(min(pred_cnt[t], gold_cnt[t]) for t in pred_cnt.keys() & gold_cnt.keys())

    precision = common / sum(pred_cnt.values()) if pred_cnt else 0.0
    recall = common / sum(gold_cnt.values()) if gold_cnt else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


# -------------------------
# Hallucination detection
# -------------------------
def ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def detect_hallucination(pred: str, evidence: str) -> Tuple[bool, List[str]]:
    """
    Detect hallucinations via n-gram grounding.
    - Use bigrams + trigrams
    - Ignore stopwords and short tokens
    """
    pred_tokens = [t for t in tokenize(pred) if t not in STOPWORDS]
    ev_tokens = [t for t in tokenize(evidence) if t not in STOPWORDS]

    ev_set = set(ev_tokens)
    ev_ngrams = set(ngrams(ev_tokens, 2) + ngrams(ev_tokens, 3))

    suspects = []
    for t in pred_tokens:
        if len(t) < 3:  # ignore very short
            continue
        if t not in ev_set:
            suspects.append(t)

    pred_ngrams = ngrams(pred_tokens, 2) + ngrams(pred_tokens, 3)
    for ng in pred_ngrams:
        if ng not in ev_ngrams:
            suspects.append(ng)

    suspects = list(set(suspects))
    return (len(suspects) > 0), suspects[:20]


# -------------------------
# Scoring rubric
# -------------------------
def score_question(res: dict) -> float:
    score = 0.0
    if res["exact_match"]:
        score += 1.0
    score += res["f1"]  # up to +1
    if res["coverage_ok"]:
        score += 0.5
    if not res["hallucination"]:
        score += 0.5
    return round(score, 3)


# -------------------------
# Evaluator
# -------------------------
def evaluate_question(question: str, expected: str, docs_folder: str) -> dict:
    # retrieve evidence
    chunks, meta = load_and_chunk_docs(docs_folder)
    embedder, index, _ = build_index(chunks)
    retrieved = retrive(question, embedder, index, chunks, meta, top_k=10)
    retrieved = rerank(question, retrieved, top_m=6)

    bullets = []
    for i, (chunk_text, _, _) in enumerate(retrieved, 1):
        b = extractive_bullet(chunk_text, question, embedder)
        labeled = f"(B{i}) {b[2:].strip()}" if b.startswith("- ") else f"(B{i}) {b.strip()}"
        bullets.append(labeled)
    unique_bullets = dedup_bullets_semantic(bullets, threshold=0.92, embedder=embedder)
    evidence_text = " ".join([re.sub(r"^\(B\d+\)\s*", "", b) for b in unique_bullets])

    # model answer
    predicted, provenance = advanced_rag(question, docs_folder=docs_folder, top_k=10, top_m=6, use_rerank=True)
    # parts = model_out.split("\n\nSources:")
    # predicted = parts[0].strip()
    # provenance = ("Sources:" + parts[1].strip()) if len(parts) > 1 else ""

    # metrics
    exact = (normalize_text(expected) == normalize_text(predicted))
    p, r, f1 = compute_f1(tokenize(predicted), tokenize(expected))
    coverage = r >= 0.99
    hallucinated, suspects = detect_hallucination(predicted, evidence_text)
    q_score = score_question({
        "exact_match": exact,
        "f1": f1,
        "coverage_ok": coverage,
        "hallucination": hallucinated
    })

    return {
        "question": question,
        "expected": expected,
        "predicted": predicted,
        "provenance": provenance,
        "exact_match": exact,
        "precision": p,
        "recall": r,
        "f1": f1,
        "coverage_ok": coverage,
        "hallucination": hallucinated,
        "hallucinated_tokens_sample": suspects,
        "score": q_score,
        "num_bullets_used": len(unique_bullets),
    }


def run_eval(docs_folder: str = "/home/yesituser/yesitprojects/venv/isoTracker_requirement_AI/advanced_RAG/docs"):
    results = []
    total_score = 0.0

    for qa in EVAL_QUESTIONS:
        q, expected = qa["question"], qa["expected"]
        print("\n====================")
        print("Q:", q)
        res = evaluate_question(q, expected, docs_folder)
        results.append(res)
        total_score += res["score"]

        # print result
        print("Predicted:", res["predicted"])
        print("Expected :", res["expected"])
        print(f"Exact: {res['exact_match']}, F1={res['f1']:.3f}, Score={res['score']}")
        print(f"Coverage OK: {res['coverage_ok']}, Hallucination: {res['hallucination']}")
        if res["hallucination"]:
            print("Hallucinated sample:", res["hallucinated_tokens_sample"])
        print("Provenance:", res["provenance"])

    avg_score = total_score / len(results) if results else 0.0
    print("\n====================")
    print("Final Diagnostic Summary")
    print("--------------------")
    print(f"Questions evaluated: {len(results)}")
    print(f"Average rubric score: {avg_score:.2f} / 3.0")
    print(f"Exact matches: {sum(r['exact_match'] for r in results)} / {len(results)}")
    print(f"Avg F1: {sum(r['f1'] for r in results)/len(results):.3f}")
    print(f"Coverage OK: {sum(r['coverage_ok'] for r in results)} / {len(results)}")
    print(f"Hallucinations: {sum(r['hallucination'] for r in results)} / {len(results)}")

    with open("eval_results_module8.json", "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2)
    print("\nWrote eval_results_module8.json")


if __name__ == "__main__":
    run_eval("/home/yesituser/yesitprojects/venv/isoTracker_requirement_AI/advanced_RAG/docs")
