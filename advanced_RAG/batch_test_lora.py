import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# -----------------------------
# Config
# -----------------------------
LORA_WEIGHTS = "./lora-requirement-master"
OUTPUT_FILE = "lora_outputs.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load PEFT config and base model
# -----------------------------
print("Loading LoRA adapter...")
peft_config = PeftConfig.from_pretrained(LORA_WEIGHTS)
BASE_MODEL = peft_config.base_model_name_or_path

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS)
model.eval()

# -----------------------------
# Load Dataset
# -----------------------------
dataset = load_dataset("json", data_files="train_requirements.jsonl", split="train")

# -----------------------------
# Run Inference and Save Outputs
# -----------------------------
print("Generating outputs...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i, example in enumerate(dataset):
        prompt = f"{example['instruction']}\n\n{example['input']}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        decoded_output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        f.write(f"=== EXAMPLE {i+1} ===\n")
        f.write(f"PROMPT:\n{prompt}\n\n")
        f.write(f"OUTPUT:\n{decoded_output}\n\n\n")

print(f"✅ All outputs saved to {OUTPUT_FILE}")
