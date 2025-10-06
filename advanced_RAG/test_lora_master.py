import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# -----------------------------
# Configs
# -----------------------------
import re

def clean_headings(text):
    text = re.sub(r"#+\s*\w.*\n?", "", text)      # remove markdown headings
    text = re.sub(r"\n{2,}", "\n", text)          # collapse blank lines
    return text.strip()

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_ADAPTER = "./lora-requirement-master"

# -----------------------------
# Load Model + Adapter
# -----------------------------
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Attaching LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Inference Helper
# -----------------------------
def generate_requirement_response(context, question):
    instruction = (
    "You are a Requirement Trainer. "
    "Explain the requirement in a clear, narrative, paragraph style — "
    "do NOT use headings, bullet points, or section titles. "
    "Speak naturally as if mentoring a new analyst."
)
    prompt = f"{instruction}\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = clean_headings(response)
    print("\n🧠 Cleaned Narrative:\n", response)

# -----------------------------
# Test Cases
# -----------------------------
# ✅ Positive / Clear Requirement
generate_requirement_response(
    "The HR system must allow employees to update personal data and generate payslips monthly.",
    "Tell me about this module."
)

# ⚠️ Conflictive Requirement
generate_requirement_response(
    "The data retention policy mentions 3 years in one paragraph and 5 years in another.",
    "Are there any conflicts or ambiguities?"
)

# ❌ Missing Info Case
generate_requirement_response(
    "The fleet tracking software must show real-time vehicle locations and alert for route deviations.",
    "What is the project deadline?"
)
