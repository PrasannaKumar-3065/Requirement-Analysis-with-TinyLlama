# validate_narrative_dataset.py
import json
from pathlib import Path

path = Path("train_requirements_narrative.jsonl")

invalid, short = [], []
for i, line in enumerate(path.open("r", encoding="utf-8"), 1):
    try:
        obj = json.loads(line)
        if len(obj.get("output", "").split()) < 25:
            short.append(i)
    except json.JSONDecodeError:
        invalid.append(i)

print(f"Total lines: {i}")
print(f"Invalid JSON: {len(invalid)} lines → {invalid[:10]}")
print(f"Short narratives (<25 words): {len(short)} lines → {short[:10]}")
