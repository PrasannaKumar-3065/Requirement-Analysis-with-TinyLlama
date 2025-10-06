import json

INPUT_FILE = "refinement_ready.jsonl"
OUTPUT_FILE = "train_weighted.jsonl"

def compute_trainer_score(entry):
    fluency = entry.get("fluency", 0)
    rougeL = entry.get("rougeL", 0)
    bertscore_f1 = entry.get("bertscore_f1", 0)
    has_conflict_terms = entry.get("has_conflict_terms", False)

    score = (
        0.3 * fluency +
        0.4 * bertscore_f1 +
        0.2 * rougeL +
        (0.1 if has_conflict_terms else 0)
    )
    return round(score, 3)

def assign_weight(score):
    if score >= 0.82:
        return "high"
    elif score >= 0.70:
        return "medium"
    else:
        return "low"

def build_weighted_dataset():
    weighted_records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        for line in infile:
            try:
                record = json.loads(line)
                score = compute_trainer_score(record)
                record["trainer_score"] = score
                record["weight"] = assign_weight(score)
                weighted_records.append(record)
            except json.JSONDecodeError:
                print("⚠️ Skipping invalid line")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for r in weighted_records:
            json.dump(r, outfile, ensure_ascii=False)
            outfile.write("\n")

    print(f"✅ Weighted dataset created: {OUTPUT_FILE}")
    print(f"📊 Records processed: {len(weighted_records)}")

if __name__ == "__main__":
    build_weighted_dataset()
