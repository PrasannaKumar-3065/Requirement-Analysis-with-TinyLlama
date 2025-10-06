# build_index.py
import os, json
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ===============================
# CONFIGURATION
# ===============================
DOCUMENTS_DIR = "docs"
INDEX_DIR = "vector_index"
os.makedirs(INDEX_DIR, exist_ok=True)

# ===============================
# SIMPLE DOCUMENT LOADER
# (Docling-ready stub)
# ===============================
def load_documents():
    """
    Loads text from .txt, .pdf, .md, and .docx files.
    Docling support can be added later here.
    """
    from pathlib import Path
    import docx
    from PyPDF2 import PdfReader

    docs = []
    for file in Path(DOCUMENTS_DIR).glob("*"):
        text = ""
        if file.suffix.lower() == ".txt":
            text = file.read_text(encoding="utf-8", errors="ignore")
        elif file.suffix.lower() == ".pdf":
            try:
                pdf = PdfReader(str(file))
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            except Exception:
                print(f"⚠️ Failed to read PDF: {file}")
        elif file.suffix.lower() == ".docx":
            try:
                doc = docx.Document(str(file))
                text = "\n".join([p.text for p in doc.paragraphs])
            except Exception:
                print(f"⚠️ Failed to read DOCX: {file}")
        if text.strip():
            docs.append({"source": file.name, "text": text.strip()})
    return docs


print("📚 Loading documents from", DOCUMENTS_DIR)
documents = load_documents()
print(f"✅ Loaded {len(documents)} documents.")

# ===============================
# CHUNKING (Basic Sentence Split)
# ===============================
from textwrap import wrap
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def chunk_text(text):
    words = text.split()
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        yield " ".join(words[i : i + CHUNK_SIZE])

chunks = []
for doc in documents:
    for i, chunk in enumerate(chunk_text(doc["text"])):
        chunks.append({"text": chunk, "source": doc["source"], "chunk_id": i})
print(f"🧩 Produced {len(chunks)} chunks.")

# ===============================
# EMBEDDINGS
# ===============================
print("🔹 Loading SentenceTransformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode([c["text"] for c in chunks], show_progress_bar=True, convert_to_numpy=True)
print(f"✅ Embeddings generated (dim={embeddings.shape[1]})")

# ===============================
# BUILD FAISS INDEX
# ===============================
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
print(f"✅ FAISS index built: {index.ntotal} vectors.")

# Save chunk metadata
with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)
print("✅ Saved chunk metadata.")

# ===============================
# OPTIONAL BM25 LEXICAL INDEX
# ===============================
tokenized_corpus = [c["text"].split() for c in chunks]
bm25 = BM25Okapi(tokenized_corpus)
with open(os.path.join(INDEX_DIR, "bm25.json"), "w", encoding="utf-8") as f:
    json.dump({"tokenized": tokenized_corpus}, f)
print("✅ BM25 index saved.")

print("\n🎯 Indexing complete.")
