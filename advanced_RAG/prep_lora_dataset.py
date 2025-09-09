import json
import random

def add_noise(bullets):
    noise_pool = [
        "(B?) Risk: vendor support may delay integration.",
        "(B?) Staff noted cafeteria renovation is off-schedule.",
        "(B?) Reminder: company holiday next week.",
        "(B?) Placeholder text with no real info.",
    ]
    n = random.randint(1, 2)
    return bullets + random.sample(noise_pool, n)

examples = []

# Clean
examples.append({
    "question": "Who is the client?",
    "bullets": ["(B1) The client for this project is ACME Corp."],
    "answer": "The client is ACME Corp (B1)."
})

# Messy
examples.append({
    "question": "Who is the client?",
    "bullets": add_noise([
        "(B1) Client confirmed mobile app is top priority for warehouse staff.",
        "(B2) The client for this project is ACME Corp.",
    ]),
    "answer": "The client is ACME Corp (B2). Other bullets describe priorities or risks."
})

# Conflict
examples.append({
    "question": "What is the project deadline?",
    "bullets": [
        "(B1) Delivery is expected no later than November 2025.",
        "(B2) Deadline for the project is October 2025.",
    ],
    "answer": "There is a conflict: October 2025 (B2) vs November 2025 (B1). The preferred deadline is October 2025."
})

# Performance
examples.append({
    "question": "What are the performance requirements?",
    "bullets": add_noise([
        "(B1) The system must handle up to 500 concurrent users without performance degradation.",
        "(B2) Performance requirement was highlighted: 2-second response time under heavy load.",
    ]),
    "answer": "The system must handle 500 concurrent users (B1) and provide a 2-second response time under heavy load (B2)."
})

# Expand with noisy variations
dataset = []
for e in examples:
    for _ in range(3):
        noisy_bullets = add_noise(e["bullets"])
        dataset.append({
            "instruction": e["question"],
            "input": "\n".join(noisy_bullets),
            "output": e["answer"]
        })

with open("train_data_lora.json", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")

print(f"✅ Wrote {len(dataset)} examples to train_data_lora.json")