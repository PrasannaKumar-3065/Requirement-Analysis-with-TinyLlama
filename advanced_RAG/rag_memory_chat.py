import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from memory_manager import MemoryManager
from uuid import uuid4

# -----------------------------
# CONFIG
# -----------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./lora-policy-aligned"
CHUNK_STORE_PATH = "vector_index/chunks.json"
FAISS_PATH = "vector_index/faiss.index"

# -----------------------------
# INITIALIZE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMBED_MODEL)
index = faiss.read_index(FAISS_PATH)
with open(CHUNK_STORE_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"✅ Loaded {len(chunks)} chunks from store.")

# Load LoRA model
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", max_new_tokens=250)

# Memory manager
memory = MemoryManager()
session_id = str(uuid4())[:8]  # short session id
print(f"🧠 New Chat Session: {session_id}")

# -----------------------------
# FUNCTIONS
# -----------------------------
def retrieve_context(query, top_k=3):
    qvec = embedder.encode([query])
    scores, ids = index.search(qvec.astype("float32"), top_k)
    results = []
    for i, idx in enumerate(ids[0]):
        if idx < len(chunks):
            results.append(chunks[idx])
    return results

def build_prompt(query, history, retrieved):
    history_text = "\n".join([f"{r.upper()}: {c}" for r, c in history])
    retrieved_text = "\n".join([f"- {r['text']}" for r in retrieved])
    prompt = f"""
You are RequirementMaster AI.
Below is the conversation so far:
{history_text}

Retrieved project context:
{retrieved_text}

Now, respond to the new user question:
User: {query}
AI:
"""
    return prompt.strip()

# -----------------------------
# MAIN LOOP
# -----------------------------
print("\n🤖 RAG Chat with Memory (LoRA Enhanced)")
print("Type 'exit' to quit or 'clear' to reset memory.\n")

while True:
    query = input("🧠 You: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting chat.")
        break
    if query.lower() == "clear":
        memory.clear_memory(session_id)
        print("🧹 Memory cleared for this session.")
        continue

    # Retrieve and build context
    retrieved = retrieve_context(query)
    history = memory.get_history(session_id, limit=5)
    prompt = build_prompt(query, history, retrieved)

    # Generate
    response = generator(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
    response = response.split("AI:")[-1].strip()

    # Display
    print("\n💬 AI:", response, "\n" + "-" * 70)

    # Save to memory
    memory.add_message(session_id, "user", query)
    memory.add_message(session_id, "ai", response)
