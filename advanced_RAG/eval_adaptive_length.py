# eval_adaptive_length.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

BASE_MODEL = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ADAPTER_DIR = os.getenv("LORA_ADAPTIVE_OUT", "./lora-adaptive-length")
DEVICE = 0 if torch.cuda.is_available() else -1

print("Loading model + adapter...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# load base (with same dtype/device_map as training)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
# load LoRA adapter (PEFT)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)

gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=200,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)

def run_example(context, question, expected_length):
    prompt = (
        "You are a Requirement Trainer. Analyze the following in the specified style.\n\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Expected length: {expected_length}\n\nAnswer:\n"
    )
    out = gen(prompt)[0]["generated_text"]
    # post-process to strip prompt prefix
    generated = out.split("Answer:")[-1].strip()
    print("\n--- PROMPT ---")
    print(prompt)
    print("\n--- GENERATED ---")
    print(generated)
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    # quick interactive checks
    examples = [
        ("The HR system must allow employees to update personal data and generate payslips monthly.", "Tell me about this module.", "narrative"),
        ("The fleet tracking software must show real-time vehicle locations and alert for route deviations.", "What is the project deadline?", "brief"),
        ("The API must return data within 200 ms, while a parallel spec allows up to 1 second latency.", "What is the performance requirement?", "analytical"),
    ]
    for ctx, q, length in examples:
        run_example(ctx, q, length)

    # If you want manual testing uncomment below:
    # while True:
    #     ctx = input("Context (empty to quit): ").strip()
    #     if not ctx:
    #         break
    #     q = input("Question: ").strip()
    #     length = input("Expected length (brief|analytical|narrative): ").strip() or "analytical"
    #     run_example(ctx, q, length)
