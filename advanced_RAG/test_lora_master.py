from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_WEIGHTS = "./lora-requirement-master"

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0
)

prompt = """You are a Requirement Master. Analyze the requirement.

Context:
Project deadline: October 2025.
Stakeholders mentioned an alternative deadline of November 2025.

Question: What is the project deadline?

Output:
### Summary:"""   # cue for continuation

print("\n====================")
print("MODEL OUTPUT:\n")

output = pipe(
    prompt,
    max_new_tokens=600,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.2
)[0]["generated_text"]

# Just show the model’s continuation (not the prompt)
answer = output[len(prompt):].strip()
print(answer+"\n------------")
print(output)
