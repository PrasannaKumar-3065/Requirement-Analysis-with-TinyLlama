from transformers import pipeline

ner = pipeline("token-classification", model="dslim/bert-base-NER")

text = "The new payment system must be delivered by October 2025 for the client ACME Corp."
result = ner(text)

print("Entities:", result)
