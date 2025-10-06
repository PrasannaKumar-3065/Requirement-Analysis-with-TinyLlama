# build_contextual_grounding_dataset.py
import json
from pathlib import Path
import random

OUTPUT_PATH = "contextual_grounding_dataset.jsonl"
NUM_SAMPLES = 100

styles = ["brief", "analytical", "narrative"]

contexts = [
    (
        "The HR system must allow employees to update personal data and generate payslips monthly.",
        "Tell me about this module.",
        {
            "brief": "This module lets employees update personal data and view monthly payslips.",
            "analytical": "The HR module provides editable employee profiles and automates monthly payslip generation, ensuring compliance and transparency.",
            "narrative": "The HR system’s module empowers employees to maintain accurate personal data while automatically producing monthly payslips, reducing manual effort and ensuring payroll clarity."
        },
    ),
    (
        "The fleet tracking software must show real-time vehicle locations and alert for route deviations.",
        "What is the key function of this software?",
        {
            "brief": "It tracks vehicle locations and alerts on route deviations.",
            "analytical": "The system ensures operational visibility by monitoring real-time locations and triggering alerts for deviation events.",
            "narrative": "This fleet tracking system offers full visibility of each vehicle’s real-time position while sending immediate alerts whenever a route deviation occurs, improving route reliability."
        },
    ),
    (
        "The API must return data within 200 ms, while a parallel spec allows up to 1 second latency.",
        "What is the performance requirement?",
        {
            "brief": "The API must respond within 200 ms.",
            "analytical": "The API performance requirement is to return data in 200 ms, although a conflicting specification permits up to 1 second latency.",
            "narrative": "Under the system’s strict performance constraint, the API is expected to deliver responses within 200 milliseconds, though an alternative spec allows up to one second, indicating a need for clarification."
        },
    ),
    (
        "The billing system must allow reversible adjustments within 30 days, log all changes, and enforce role separation.",
        "What is the main requirement?",
        {
            "brief": "The billing system supports reversible changes within 30 days and logs all actions.",
            "analytical": "The system mandates reversible billing adjustments within 30 days while maintaining change logs and role-based access for accountability.",
            "narrative": "This billing module enables users to reverse adjustments within 30 days, ensuring all modifications are logged and each action adheres to defined user roles, promoting transparency."
        },
    ),
]

print(f"Generating {NUM_SAMPLES} samples...")
data = []

for _ in range(NUM_SAMPLES):
    ctx, q, outputs = random.choice(contexts)
    style = random.choice(styles)
    example = {
        "instruction": "You are a Requirement Trainer. Answer in the requested style and stay grounded in the given context.",
        "input": f"Context: {ctx}\nQuestion: {q}\nExpected style: {style}",
        "output": outputs[style]
    }
    data.append(example)

Path(OUTPUT_PATH).write_text("\n".join(json.dumps(x) for x in data), encoding="utf-8")
print(f"✅ Saved {len(data)} samples to {OUTPUT_PATH}")
