import json
import random
from tqdm import tqdm

# Seed for consistent generation
random.seed(42)

contexts = [
    "The HR system must allow employees to update personal data and generate payslips monthly.",
    "The API must return data within 200 ms, while another spec allows up to 1 second latency.",
    "The data retention policy mentions 3 years in one paragraph and 5 years in another.",
    "The analytics dashboard must display live KPIs from sales and operations in one unified view.",
    "The CI pipeline must finish tests within 10 minutes and block merges if coverage falls below 80%.",
    "The inventory management module must synchronize stock levels across warehouses every hour.",
    "All personal data for EU customers must be handled according to GDPR and stored encrypted at rest.",
    "The backup system must run every 12 hours and be recoverable within 2 hours.",
]

questions = {
    "brief": [
        "What is the project deadline?",
        "Who is the client?",
        "What is the company name?",
        "Is the technology stack specified?",
    ],
    "analytical": [
        "Are there any conflicts or ambiguities?",
        "What compliance issues could arise?",
        "What are the dependencies?",
    ],
    "narrative": [
        "Tell me about this module.",
        "Explain how this system operates.",
        "Describe its overall purpose and rationale."
    ],
}

outputs = {
    "brief": [
        "The project deadline is not specified in the requirement.",
        "The client name is not mentioned in the given context.",
        "The company name is not stated.",
        "The technology stack is not specified in the requirement details."
    ],
    "analytical": [
        "There is a conflict between retention durations, requiring clarification.",
        "Ambiguities exist regarding timing and dependency alignment between services.",
        "Compliance issues could arise if encryption is not enforced at rest."
    ],
    "narrative": [
        "This module automates HR operations by allowing employees to update their personal information and generating monthly payslips to maintain transparency.",
        "The system ensures continuous monitoring of fleet operations, providing real-time updates and deviation alerts for improved route management.",
        "This dashboard aggregates performance metrics from multiple departments, enabling data-driven strategic decision-making."
    ],
}

with open("adaptive_length_dataset.jsonl", "w", encoding="utf-8") as f:
    for _ in tqdm(range(100)):
        context = random.choice(contexts)
        style = random.choice(["brief", "analytical", "narrative"])
        question = random.choice(questions[style])
        output = random.choice(outputs[style])
        record = {
            "instruction": "You are a Requirement Trainer. Analyze the following in the specified style.",
            "input": f"Context: {context}\nQuestion: {question}\nExpected length: {style}",
            "output": output
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("✅ adaptive_length_dataset.jsonl generated successfully.")
