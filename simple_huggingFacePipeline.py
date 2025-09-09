from transformers import pipeline

# Text-generation pipeline
generator = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

prompt = "Explain requirement analysis in one simple sentence."
result = generator(prompt, max_new_tokens=50, temperature=0.7)

print("Generated:", result[0]['generated_text'])
