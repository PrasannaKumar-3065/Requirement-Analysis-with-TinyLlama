# plot_policy_alignment_results.py
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------
# CONFIG
# -----------------------------
reports = {
    "v7_hybrid": "v7_hybrid_policy_eval_report.json",
    "policy_aligned": "policy_aligned_policy_eval_report.json",
}

metric_names = ["ROUGE-L", "BERTScore-F1"]
metrics = {}

# -----------------------------
# LOAD REPORTS
# -----------------------------
for name, path in reports.items():
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {path}")
        continue
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Normalize keys
        entry = {}
        for k, v in data.items():
            k_norm = k.lower().replace("-", "").replace("_", "")
            if "rouge" in k_norm:
                entry["ROUGE-L"] = v
            elif "bert" in k_norm:
                entry["BERTScore-F1"] = v
        metrics[name] = entry
        print(f"✅ Loaded metrics for {name}: {entry}")

# -----------------------------
# PREPARE DATA
# -----------------------------
v7_vals = [metrics.get("v7_hybrid", {}).get(m, 0) for m in metric_names]
aligned_vals = [metrics.get("policy_aligned", {}).get(m, 0) for m in metric_names]

x = np.arange(len(metric_names))
width = 0.35

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(8, 5))
bars1 = plt.bar(x - width/2, v7_vals, width, label="v7 Hybrid")
bars2 = plt.bar(x + width/2, aligned_vals, width, label="Policy-Aligned")

plt.xticks(x, metric_names)
plt.ylabel("Score")
plt.title("Policy Alignment Impact on LoRA Evaluation Metrics")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend()

# Annotate bar values
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2, height + 0.01,
            f"{height:.3f}", ha="center", va="bottom", fontsize=9
        )

plt.tight_layout()
plt.show()
