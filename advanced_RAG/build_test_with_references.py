# build_test_with_references.py
import json

samples = [
    {
        "instruction": "You are a Requirement Trainer.",
        "input": "Context: The HR system must allow employees to update personal data and generate payslips monthly.\nQuestion: Tell me about this module.",
        "output": "This HR module allows employees to update their personal data and automatically generates monthly payslips, ensuring transparency."
    },
    {
        "instruction": "You are a Requirement Trainer.",
        "input": "Context: The API must return data within 200 ms while another spec allows 1 second latency.\nQuestion: What is the performance requirement?",
        "output": "The performance requirement is that the API must respond within 200 ms, though another specification allows up to 1 second latency, indicating a conflict."
    },
    {
        "instruction": "You are a Requirement Trainer.",
        "input": "Context: The system must log all reversible adjustments within 30 days and enforce role separation.\nQuestion: What is required?",
        "output": "The system should support reversible adjustments within 30 days, ensure logs of every change, and enforce separation of duties between initiators and approvers."
    },
    {
        "instruction": "You are a Requirement Trainer.",
        "input": "Context: The inventory management module must synchronize stock every hour.\nQuestion: What is the sync interval?",
        "output": "Stock synchronization occurs every hour across all warehouses."
    },
    {
        "instruction": "You are a Requirement Trainer.",
        "input": "Context: The project must comply with GDPR and encrypt personal data at rest.\nQuestion: What are the compliance requirements?",
        "output": "The system must comply with GDPR and ensure that personal data is encrypted at rest and during transmission."
    },
]

with open("test_requirements_ref.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")

print("✅ test_requirements_ref.jsonl created with reference outputs.")
