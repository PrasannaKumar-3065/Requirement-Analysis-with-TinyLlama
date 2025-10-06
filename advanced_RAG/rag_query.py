# rag_query_with_lora.py
# RAG pipeline using FAISS + LoRA fine-tuned TinyLlama

import os
import torch
import faiss
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# -------------------------------
# CONFIG
# -------------------------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_ADAPTER_PATH = "./lora-policy-aligned"     # your LoRA fine-tuned weights
FAISS_INDEX_PATH = "vector_index/faiss.index"
CHUNK_STORE_PATH = "vector_index/chunks.json"
BM25_INDEX_PATH = "vector_index/bm25.json"

# -------------------------------
# LOAD RETRIEVAL COMPONENTS
# -------------------------------
print("🔹 Loading embedding model and FAISS index...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)

# Load chunk texts
with open(CHUNK_STORE_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load BM25 lexical index (optional)
# Load BM25 lexical index (optional)
if os.path.exists(BM25_INDEX_PATH):
    try:
        with open(BM25_INDEX_PATH, "r", encoding="utf-8") as f:
            bm25_data = json.load(f)

        # ✅ Handle both possible formats
        if isinstance(bm25_data, dict):
            if "corpus" in bm25_data:
                corpus = bm25_data["corpus"]
            elif "documents" in bm25_data:
                corpus = bm25_data["documents"]
            else:
                corpus = list(bm25_data.values())
        elif isinstance(bm25_data, list):
            corpus = bm25_data
        else:
            raise ValueError("Invalid BM25 format")

        bm25 = BM25Okapi(corpus)
        print("✅ BM25 index loaded successfully.")
    except Exception as e:
        print(f"⚠️ Failed to load BM25 index ({e}). Skipping BM25...")
        bm25 = None
else:
    bm25 = None

print(f"Loaded {len(chunks)} chunks from store.\n")

# -------------------------------
# LOAD LoRA MODEL FOR GENERATION
# -------------------------------
print("⚙️ Loading base model + LoRA adapter...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Attach LoRA adapter
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
model.eval()

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=300,
    temperature=0.3,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

print("✅ LoRA-based generator initialized.\n")

# -------------------------------
# RAG RETRIEVAL FUNCTION
# -------------------------------
def retrieve_context(query, top_k=3):
    """Retrieve the top_k relevant document chunks based on embeddings + BM25 fusion."""
    # Step 1: Embedding-based retrieval
    query_vec = embedder.encode([query])
    scores, indices = index.search(np.array(query_vec, dtype=np.float32), top_k)
    
    faiss_results = [(chunks[i], float(scores[0][idx])) for idx, i in enumerate(indices[0])]

    # Step 2: BM25 lexical rerank (if available)
    if bm25:
        bm25_scores = bm25.get_scores(query.split())
        bm25_pairs = [(chunks[i], float(bm25_scores[i])) for i in range(len(chunks))]
        bm25_pairs = sorted(bm25_pairs, key=lambda x: x[1], reverse=True)[:top_k]
        # merge scores
        combined = {}
        for txt, sc in faiss_results + bm25_pairs:
            combined[txt] = combined.get(txt, 0) + sc
        reranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
    else:
        return faiss_results

# -------------------------------
# MAIN LOOP
# -------------------------------
print("🤖 RAG Query Engine Ready (LoRA version). Type your queries below.\n")

while True:
    try:
        query = input("🧠 Query: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("👋 Exiting RAG engine.")
            break

        results = retrieve_context(query, top_k=3)

        print("\n📚 Retrieved Contexts:")
        context_text = ""
        for i, (text, score) in enumerate(results, 1):
            print(f"\n[{i}] (score={score:.4f})")
            print(text)
            context_text += f"\n\n{text}"

        # Construct generation prompt
        prompt = f"""
You are a Requirement Master AI. Analyze the question based on the following context.

Context:
{context_text}

Question:
{query}

Answer:
"""

        output = generator(prompt, max_new_tokens=200)[0]["generated_text"]
        answer = output.split("Answer:")[-1].strip() if "Answer:" in output else output.strip()

        print("\n💬 AI Answer:\n" + "-"*70)
        print(answer)
        print("-"*70 + "\n")

    except KeyboardInterrupt:
        print("\n👋 Interrupted. Exiting.")
        break
