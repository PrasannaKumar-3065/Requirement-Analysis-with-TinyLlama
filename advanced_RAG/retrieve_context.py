# retrieve_context.py
import json
import faiss
import torch
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
FAISS_INDEX_PATH = "vector_index/faiss.index"
CHUNK_STORE_PATH = "vector_index/chunks.json"
BM25_INDEX_PATH = "vector_index/bm25.json"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# -----------------------------
# LOAD MODELS AND INDEXES
# -----------------------------
print("Loading embedding model and FAISS index...")
embedder = SentenceTransformer(EMBED_MODEL_NAME)

index = faiss.read_index(FAISS_INDEX_PATH)

with open(CHUNK_STORE_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Optional BM25 index
try:
    with open(BM25_INDEX_PATH, "r", encoding="utf-8") as f:
        bm25_data = json.load(f)
        bm25 = BM25Okapi(bm25_data["tokenized"])
    print("BM25 index loaded successfully.")
except FileNotFoundError:
    bm25 = None
    print("BM25 index not found. Skipping lexical retrieval.")

print(f"Loaded {len(chunks)} chunks from store.")

# -----------------------------
# RETRIEVAL FUNCTION
# -----------------------------
def retrieve(query, top_k=5, alpha=0.6):
    """
    alpha controls weight balance between FAISS (semantic) and BM25 (lexical)
    alpha = 1.0 => semantic only
    alpha = 0.0 => lexical only
    """
    # ---- 1. FAISS retrieval ----
    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, top_k)
    faiss_hits = [(i, float(D[0][rank])) for rank, i in enumerate(I[0])]

    # ---- 2. BM25 retrieval ----
    bm25_hits = []
    if bm25:
        scores = bm25.get_scores(query.split())
        top_bm25 = np.argsort(scores)[::-1][:top_k]
        bm25_hits = [(i, float(scores[i])) for i in top_bm25]

    # ---- 3. Merge results ----
    combined = {}
    for i, s in faiss_hits:
        combined[i] = combined.get(i, 0) + alpha * s
    for i, s in bm25_hits:
        combined[i] = combined.get(i, 0) + (1 - alpha) * s

    # ---- 4. Rank and return ----
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = [{"text": chunks[i]["text"], "score": score} for i, score in ranked]
    seen = set()
    unique_results = []
    for text, score in results:
        if text not in seen:
            unique_results.append((text, score))
            seen.add(text)
    return unique_results

# -----------------------------
# CLI INTERFACE
# -----------------------------
if __name__ == "__main__":
    print("\n🔍 Retrieval Ready! Type your query below:")
    while True:
        query = input("\n🧠 Query: ").strip()
        if not query:
            break
        results = retrieve(query, top_k=5)
        print("\n📚 Top Retrieved Chunks:")
        for rank, r in enumerate(results, 1):
            print(f"\n[{rank}] (score={r['score']:.4f})\n{r['text']}\n{'-'*60}")
