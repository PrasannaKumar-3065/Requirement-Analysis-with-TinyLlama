from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

requirement = "Hello there How may i help you."
result = classifier(requirement)

print("Classification:", result)
