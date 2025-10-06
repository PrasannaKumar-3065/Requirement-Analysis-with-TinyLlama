# evaluate_self_consistency.py
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
from statistics import mean
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./lora-requirement-master-v6"
TEST_DATA_PATH = "test_requirements.jsonl"
OUTPUT_PATH = "self_consistency_report.json"
NUM_VARIANTS = 5
DEVICE = 0 if torch.cuda.is_available() else -1

# -----------------------------
# LOAD MODEL + TOKENIZER
# -----------------------------
print(f"🔹 Loading model: {BASE_MODEL} with adapter: {ADAPTER_PATH}")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)
try:
    model.load_adapter(ADAPTER_PATH)
    model.set_adapter(ADAPTER_PATH)
    print("✅ Adapter loaded successfully.")
except Exception:
    print("⚠️ Could not load adapter (may already be merged or default). Continuing.")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")

gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=200,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)

# -----------------------------
# LOAD TEST DATA
# -----------------------------
with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

print(f"📄 Loaded {len(test_data)} test samples.")
results = []

# -----------------------------
# EVALUATION LOOP
# -----------------------------
for i, sample in enumerate(tqdm(test_data, desc="Evaluating consistency")):
    prompt = (
        "You are a Requirement Trainer. Analyze the following:\n\n"
        + sample["input"]
        + "\n\nAnswer:\n"
    )

    # Generate multiple variants
    outputs = []
    for _ in range(NUM_VARIANTS):
        out = gen(prompt, max_new_tokens=200, do_sample=True, temperature=0.8, top_p=0.9)[0]["generated_text"]
        answer = out.split("Answer:")[-1].strip()
        outputs.append(answer)

    # Compute embeddings and similarity
    embeddings = embedder.encode(outputs, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embeddings, embeddings)
    upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    consistency_score = float(upper_triangle.mean().item()) if len(upper_triangle) > 0 else 0

    results.append({
        "id": i + 1,
        "input": sample["input"],
        "variants": outputs,
        "self_consistency_score": consistency_score
    })

# -----------------------------
# AGGREGATE RESULTS
# -----------------------------
avg_score = mean([r["self_consistency_score"] for r in results])
report = {
    "adapter": ADAPTER_PATH,
    "avg_self_consistency": avg_score,
    "samples": results
}

# -----------------------------
# SAVE
# -----------------------------
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n📊 Evaluation complete.")
print(f"Average Self-Consistency: {avg_score:.4f}")
print(f"✅ Report saved to {OUTPUT_PATH}")
