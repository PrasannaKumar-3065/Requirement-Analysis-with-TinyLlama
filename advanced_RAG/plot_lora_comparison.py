# plot_lora_comparison.py
import json
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
reports = {
    "v5_contextual": "contextual_evaluation_report.json",
    "v6_self_corrective": "self_corrective_evaluation_report.json",
}

# -----------------------------
# LOAD METRICS
# -----------------------------
metrics = {}
for name, path in reports.items():
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Support both structured (summary/details) and flat files
            if "summary" in data:
                metrics[name] = data["summary"]
            elif "reports" in data:
                # fallback: nested structure like refinement_evaluation_report
                metrics[name] = data.get("improvement", {})
            else:
                metrics[name] = data
        print(f"✅ Loaded {name} metrics from {path}")
    except FileNotFoundError:
        print(f"⚠️ Missing report: {path}")
        continue

# -----------------------------
# Normalize keys
# -----------------------------
def normalize_metrics(data):
    normalized = {}
    for k, v in data.items():
        key = k.lower().replace("-", "_")
        if "rouge" in key:
            normalized["rougeL"] = float(v)
        elif "bert" in key:
            normalized["bertscore_f1"] = float(v)
        elif "style" in key:
            normalized["avg_style_score"] = float(v)
    return normalized

# Metric names we care about
metric_names = ["rougeL", "bertscore_f1", "avg_style_score"]

# Extract and normalize
v5_vals = normalize_metrics(metrics.get("v5_contextual", {}))
v6_vals = normalize_metrics(metrics.get("v6_self_corrective", {}))

# Convert to lists of floats in consistent order
v5_plot = [v5_vals.get(m, 0.0) for m in metric_names]
v6_plot = [v6_vals.get(m, 0.0) for m in metric_names]

# Debugging output
print("\n📊 Normalized Metrics:")
print("v5:", v5_plot)
print("v6:", v6_plot)

# -----------------------------
# PLOT
# -----------------------------
x = np.arange(len(metric_names))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bar1 = ax.bar(x - width/2, v5_plot, width, label="v5 (Contextual)")
bar2 = ax.bar(x + width/2, v6_plot, width, label="v6 (Self-Corrective)")

# Labels and styling
ax.set_xlabel("Metric")
ax.set_ylabel("Score")
ax.set_title("LoRA Model Comparison: Contextual (v5) vs Self-Corrective (v6)")
ax.set_xticks(x)
ax.set_xticklabels(["ROUGE-L", "BERTScore-F1", "StyleScore"])
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.6)

# Annotate bars
for b in bar1 + bar2:
    height = b.get_height()
    ax.annotate(f"{height:.3f}", xy=(b.get_x() + b.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=9)

plt.tight_layout()
plt.show()
