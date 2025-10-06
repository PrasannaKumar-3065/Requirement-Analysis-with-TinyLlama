import json

cleaned = []
with open("train_requirements_trainer_clean.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if "category" not in obj:
            obj["category"] = "neutral"
        cleaned.append(obj)

with open("train_requirements_trainer_final.jsonl", "w", encoding="utf-8") as f:
    for obj in cleaned:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"✅ Finalized {len(cleaned)} samples.")
