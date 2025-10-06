# build_policy_alignment_dataset.py
import json
from random import choice

# ---------------------------
# Source samples
# ---------------------------
base_prompts = [
    {
        "context": "The HR system must allow employees to update personal data and generate payslips monthly.",
        "question": "Tell me about this module.",
    },
    {
        "context": "The fleet tracking software must show real-time vehicle locations and alert for route deviations.",
        "question": "What are the key safety features?",
    },
    {
        "context": "The data retention policy mentions 3 years in one paragraph and 5 years in another.",
        "question": "Are there any conflicts or ambiguities?",
    },
    {
        "context": "The backup system must run every 12 hours to a secure cloud provider and be recoverable within 2 hours.",
        "question": "What are the reliability requirements?",
    },
    {
        "context": "All personal data for EU customers must be handled according to GDPR and stored encrypted at rest.",
        "question": "Are there any compliance requirements?",
    }
]

# ---------------------------
# Example completions (good vs bad)
# ---------------------------
good_outputs = [
    "This module allows employees to securely update their personal data and automatically generates monthly payslips. It improves HR efficiency and ensures accuracy.",
    "The system provides real-time tracking of vehicle positions, alerting for any route deviations to improve safety and fleet visibility.",
    "Yes, there’s a conflict — one part states 3 years while another allows 5 years of retention. The policy should specify a single consistent duration.",
    "The system must maintain 99.9% uptime, perform backups every 12 hours, and ensure recovery within 2 hours to meet reliability goals.",
    "Yes, the system must comply with GDPR, encrypt all stored data, and allow data deletion requests from EU customers."
]

bad_outputs = [
    "The module lets employees update data.",
    "Vehicles can be tracked.",
    "It’s fine.",
    "Backup runs sometimes.",
    "GDPR applies."
]

# ---------------------------
# Build dataset
# ---------------------------
records = []
for sample in base_prompts:
    record = {
        "instruction": "You are a Requirement Trainer. Analyze the requirement below.",
        "input": f"Context: {sample['context']}\nQuestion: {sample['question']}",
        "chosen": choice(good_outputs),
        "rejected": choice(bad_outputs)
    }
    records.append(record)

# Save to file
output_path = "policy_alignment_dataset.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ Policy alignment dataset built: {output_path}")
print(f"Records: {len(records)}")