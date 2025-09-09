from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BASE = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load LoRA fine-tuned adapters
model = PeftModel.from_pretrained(BASE, "./lora_model")

prompt = "Instruction: Why is requirement analysis important?\nAnswer:"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
