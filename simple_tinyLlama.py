from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Tiny model that runs on CPU
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,   # use float16 if you have GPU
    device_map="auto"
)

# Build a simple prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain requirement analysis simply."}
]

# Convert messages to model input
if hasattr(tokenizer, "apply_chat_template"):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
else:
    prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate output
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
