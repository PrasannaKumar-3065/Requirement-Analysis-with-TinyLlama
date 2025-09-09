from dotenv import load_dotenv
import os
import glob
import re
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ----------------------------
# Text helpers
# ----------------------------
def split_sentences(text: str):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def extractive_bullet(chunk_text: str, query: str, embedder) -> str:
    """Return the top 1–2 most relevant sentences from a chunk as a bullet."""
    sents = split_sentences(chunk_text)
    if not sents:
        return ""
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")[0]
    s_embs = embedder.encode(sents, convert_to_numpy=True).astype("float32")
    sims = (s_embs @ q_emb) / (np.linalg.norm(s_embs, axis=1) * np.linalg.norm(q_emb) + 1e-8)
    top_idx = np.argsort(-sims)[:2]
    top_sents = [sents[i] for i in top_idx if len(sents[i]) > 3]
    joined = " ".join(top_sents)[:350]
    return "- " + joined if joined else ""

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    """Split documents into overlapping chunks."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def load_and_chunk_docs(folder: str) -> Tuple[List[str], List[Tuple[str, int]]]:
    """Load all .txt docs, return chunks and their metadata (filename, chunk_idx)."""
    chunks, meta = [], []
    files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    for fpath in files:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        file_chunks = chunk_text(text, chunk_size=600, overlap=120)
        for i, ch in enumerate(file_chunks):
            chunks.append(ch)
            meta.append((os.path.basename(fpath), i))
    return chunks, meta

def build_index(chunks: List[str], embedder_name="all-MiniLM-L6-v2"):
    """Encode chunks into FAISS index."""
    embedder = SentenceTransformer(embedder_name)
    embeddings = embedder.encode(chunks, convert_to_numpy=True).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return embedder, index, embeddings

def retrive(query: str, embedder: SentenceTransformer, index, chunks: List[str], meta, top_k=8):
    """Retrieve top-k chunks by dense similarity."""
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, top_k)
    idxs = idxs[0]
    scores = scores[0]
    results = []
    for i, score in zip(idxs, scores):
        if 0 <= i < len(chunks):
            results.append((chunks[i], meta[i], score))
    return results

def rerank(query: str, retrieved, top_m=5, ce_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Rerank retrieved chunks with a cross-encoder."""
    if len(retrieved) <= top_m:
        return retrieved
    cross = CrossEncoder(ce_model_name)
    pairs = [(query, r[0]) for r in retrieved]
    scores = cross.predict(pairs).tolist()
    rescored = [(r[0], r[1], s) for r, s in zip(retrieved, scores)]
    rescored = sorted(rescored, key=lambda x: x[2], reverse=True)
    return rescored[:top_m]

def normalize_for_dedup(text: str) -> str:
    t = re.sub(r"^\(B\d+\)\s*", "", text).lower().strip()
    return re.sub(r"\s+", " ", t)

def dedup_bullets_semantic(bullets: List[str], threshold: float = 0.92, embedder=None) -> List[str]:
    """Remove exact + semantic duplicates among bullets."""
    if not bullets:
        return []
    seen = set()
    exact = []
    for b in bullets:
        key = normalize_for_dedup(b)
        if key not in seen:
            seen.add(key)
            exact.append(b)
    if len(exact) <= 1:
        return exact
    raw = [normalize_for_dedup(b) for b in exact]
    embs = embedder.encode(raw, convert_to_numpy=True).astype("float32")
    keep, used = [], [False] * len(exact)
    for i in range(len(exact)):
        if used[i]:
            continue
        keep.append(exact[i])
        used[i] = True
        sims = (embs @ embs[i]) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(embs[i]) + 1e-8)
        for j in range(i + 1, len(exact)):
            if not used[j] and sims[j] >= threshold:
                used[j] = True
    return keep

# ----------------------------
# Model setup (LoRA)
# ----------------------------
BASE_MODEL = os.getenv("BASE_MODEL")
LORA_ADAPTER = os.getenv("ADAPTER_PATH")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(_base, LORA_ADAPTER)

def generate_llm(question: str, bullets: List[str], max_new_tokens: int = 200) -> str:
    """Generate answer using LoRA-adapted chat model."""
    bullet_str = "\n".join([f"(B{i+1}) {b}" for i, b in enumerate(bullets)])
    messages = [
        {"role": "system",
         "content": "You are a senior requirements analyst. Copy names, numbers, and dates exactly. "
                    "Answer in 1–3 sentences. Cite bullet numbers inline (e.g., B2). Ignore irrelevant bullets; note conflicts briefly."},
        {"role": "user", "content": f"QUESTION: {question}\n\nBULLETS:\n{bullet_str}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

def fuse_answer(bullets: List[str], query: str, embedder) -> str:
    """Fuse deduped bullets into a final LoRA answer."""
    unique = dedup_bullets_semantic(bullets, threshold=0.92, embedder=embedder)
    if not unique:
        return "No relevant information found in the provided documents."
    # Strip existing labels before we relabel inside generate_llm()
    stripped = [re.sub(r"^\(B\d+\)\s*", "", b).strip() for b in unique]
    return generate_llm(query, stripped)

# ----------------------------
# Main Advanced RAG pipeline
# ----------------------------
def advanced_rag(query: str,
                 docs_folder=os.getenv("DOCUMENTS_PATH"),
                 top_k=10, top_m=6, use_rerank=True) -> Tuple[str, List[str]]:
    chunks, meta = load_and_chunk_docs(docs_folder)
    if not chunks:
        return "No document chunks found.", []

    embedder_local, index, _ = build_index(chunks)
    retrieved = retrive(query, embedder_local, index, chunks, meta, top_k)

    retrieved = rerank(query, retrieved, top_m=top_m) if use_rerank else retrieved[:top_m]

    bullets = []
    sources = []
    for i, (chunk_text, (fname, local_idx), _score) in enumerate(retrieved, 1):
        bullet = extractive_bullet(chunk_text, query, embedder_local)
        if bullet.startswith("- "):
            bullet = bullet[2:].strip()
        labeled_bullet = f"(B{i}) {bullet}"
        bullets.append(labeled_bullet)
        sources.append(f"B{i}:{fname}#chunk{local_idx}")

    answer = fuse_answer(bullets, query, embedder_local)
    return answer, sources

if __name__ == "__main__":
    qs = [
        "What is the project deadline?",
        "Who is the client?",
        "What are the performance requirements?",
    ]
    for q in qs:
        print("\n====================")
        print("Q:", q)
        ans, src = advanced_rag(q)
        print(ans)
        print("Sources:", "; ".join(src))
