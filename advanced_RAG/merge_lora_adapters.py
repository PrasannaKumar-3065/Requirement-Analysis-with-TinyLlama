# merge_lora_adapters.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, PeftModelForCausalLM

# -----------------------------
# CONFIG
# -----------------------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTERS = [
    "./lora-adaptive-length",          # v5 - contextual
    "./lora-requirement-master-v6"     # v6 - self-corrective
]
MERGED_OUTPUT = "./lora-requirement-master-v7"

# -----------------------------
# LOAD BASE + FIRST ADAPTER
# -----------------------------
print("🔹 Loading base model and first adapter...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(model, ADAPTERS[0])
print(f"✅ Loaded first adapter: {ADAPTERS[0]}")

# -----------------------------
# MERGE ADDITIONAL ADAPTERS
# -----------------------------
for adapter_path in ADAPTERS[1:]:
    print(f"🔸 Merging adapter: {adapter_path}")
    adapter_model = PeftModel.from_pretrained(model, adapter_path)
    adapter_model = adapter_model.merge_and_unload()
    model = adapter_model
print("✅ All adapters merged successfully.")

# -----------------------------
# SAVE MERGED MODEL
# -----------------------------
print(f"💾 Saving merged adapter to {MERGED_OUTPUT}")
model.save_pretrained(MERGED_OUTPUT)
tokenizer.save_pretrained(MERGED_OUTPUT)
print("✅ Fusion complete — hybrid adapter v7 ready.")
