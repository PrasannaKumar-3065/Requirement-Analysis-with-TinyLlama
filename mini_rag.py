from sentence_transformers import SentenceTransformer
import faiss 
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

embedder = SentenceTransformer('all-MiniLM-L6-v2')

with open('sample.txt', 'r') as f:
    docs = [line.strip() for line in f.readlines() if line.strip()]

doc_embeddings = embedder.encode(docs, convert_to_tensor=True)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

generator = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def rag_query(question, top_k=1):  # default to top 1
    q_embedding = embedder.encode([question], convert_to_numpy=True)

    distances, indices = index.search(q_embedding, top_k)
    retrieved_docs = [docs[idx] for idx in indices[0]]

    context = "\n".join(retrieved_docs)
    prompt = f"<s>[INST] Use the context below to answer the question concisely and only once.\n\nContext:\n{context}\n\nQuestion: {question} [/INST]"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    outputs = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # clean answer after [/INST]
    clean_output = raw_output.split("[/INST]")[-1].strip()
    return clean_output

# print(rag_query("What is the project deadline?"))
# print(rag_query("Who is the client?"))
print(rag_query("What is the performance requirement?"))