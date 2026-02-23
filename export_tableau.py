"""
Tableau Export — Consolidate all results into clean, flat CSVs
ready for Tableau dashboard import.

Output directory: results/tableau_exports/
"""

import json, csv, os, sys
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd

BASE = r"C:/Users/johnu/Desktop/BigDataProject"
OUT  = os.path.join(BASE, "results", "tableau_exports")
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print("  TABLEAU DATA EXPORT")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. Dataset Summary
# ─────────────────────────────────────────────────────────────
dataset_summary = pd.DataFrame([
    {"metric": "Raw Data Size (GB)",         "value": 1.61},
    {"metric": "Raw Rows",                   "value": 3851586},
    {"metric": "Raw Columns",                "value": 54},
    {"metric": "Subjects",                   "value": 9},
    {"metric": "Activities",                 "value": 18},
    {"metric": "Sampling Rate (Hz)",         "value": 100},
    {"metric": "Window Duration (s)",        "value": 5.0},
    {"metric": "Windowed Feature Rows",      "value": 5447},
    {"metric": "Feature Columns",            "value": 172},
    {"metric": "Min Window Fill",            "value": 0.5},
])
path = os.path.join(OUT, "dataset_summary.csv")
dataset_summary.to_csv(path, index=False)
print(f"  [1/7] dataset_summary.csv          ({len(dataset_summary)} rows)")

# ─────────────────────────────────────────────────────────────
# 2. Model Comparison (all 4 Spark models)
# ─────────────────────────────────────────────────────────────
with open(os.path.join(BASE, "data", "model_results.json")) as f:
    model_data = json.load(f)

model_df = pd.DataFrame(model_data)
model_df["framework"] = "PySpark MLlib"
model_df = model_df.rename(columns={
    "model": "model_name",
    "accuracy": "accuracy",
    "f1_weighted": "f1_weighted",
    "eval_time_s": "eval_time_s",
})
path = os.path.join(OUT, "model_comparison.csv")
model_df.to_csv(path, index=False)
print(f"  [2/7] model_comparison.csv         ({len(model_df)} rows)")

# ─────────────────────────────────────────────────────────────
# 3. Spark vs Sklearn Comparison
# ─────────────────────────────────────────────────────────────
sklearn_df = pd.read_csv(os.path.join(BASE, "results", "sklearn_comparison.csv"))
path = os.path.join(OUT, "spark_vs_sklearn.csv")
sklearn_df.to_csv(path, index=False)
print(f"  [3/7] spark_vs_sklearn.csv         ({len(sklearn_df)} rows)")

# ─────────────────────────────────────────────────────────────
# 4. Confusion Matrix (long format — already Tableau-ready)
# ─────────────────────────────────────────────────────────────
cm_df = pd.read_csv(os.path.join(BASE, "results", "confusion_matrix.csv"))
path = os.path.join(OUT, "confusion_matrix.csv")
cm_df.to_csv(path, index=False)
print(f"  [4/7] confusion_matrix.csv         ({len(cm_df)} rows)")

# ─────────────────────────────────────────────────────────────
# 5. Per-Class Metrics + Feature Importance
# ─────────────────────────────────────────────────────────────
pc_df = pd.read_csv(os.path.join(BASE, "results", "per_class_metrics.csv"))
path = os.path.join(OUT, "per_class_metrics.csv")
pc_df.to_csv(path, index=False)
print(f"  [5a/7] per_class_metrics.csv       ({len(pc_df)} rows)")

fi_df = pd.read_csv(os.path.join(BASE, "results", "feature_importance.csv"))
path = os.path.join(OUT, "feature_importance.csv")
fi_df.to_csv(path, index=False)
print(f"  [5b/7] feature_importance.csv       ({len(fi_df)} rows)")

# ─────────────────────────────────────────────────────────────
# 6. Scalability Results (strong + weak scaling combined)
# ─────────────────────────────────────────────────────────────
# Strong scaling
strong_df = pd.read_csv(os.path.join(BASE, "data", "scalability_results.csv"))
strong_df["scaling_type"] = "strong"
strong_df["n_cores"] = strong_df["cores"].str.extract(r"\[(\d+)\]").astype(int)

# Compute speedup vs local[1] for each fraction
baseline_times = strong_df[strong_df["n_cores"] == 1].set_index("fraction")["train_sec"]
strong_df["speedup"] = strong_df.apply(
    lambda r: round(baseline_times.get(r["fraction"], r["train_sec"]) / r["train_sec"], 2)
    if r["train_sec"] > 0 else 0.0, axis=1
)

# Weak scaling
weak_df = pd.read_csv(os.path.join(BASE, "results", "weak_scaling.csv"))
weak_df["scaling_type"] = "weak"
if "n_cores" not in weak_df.columns:
    weak_df["n_cores"] = weak_df["cores"].str.extract(r"\[(\d+)\]").astype(int)
base_time_weak = weak_df.iloc[0]["train_sec"]
weak_df["speedup"] = round(base_time_weak / weak_df["train_sec"], 2)

# Combine
common_cols = ["scaling_type", "cores", "n_cores", "fraction", "rows", "train_rows",
               "train_sec", "accuracy", "f1_weighted", "speedup"]
scaling_df = pd.concat([
    strong_df[common_cols],
    weak_df[common_cols],
], ignore_index=True)

path = os.path.join(OUT, "scaling_results.csv")
scaling_df.to_csv(path, index=False)
print(f"  [6/7] scaling_results.csv          ({len(scaling_df)} rows)")

# ─────────────────────────────────────────────────────────────
# 7. Stability Results
# ─────────────────────────────────────────────────────────────
stab_df = pd.read_csv(os.path.join(BASE, "results", "stability_results.csv"))
path = os.path.join(OUT, "stability_results.csv")
stab_df.to_csv(path, index=False)
print(f"  [7/7] stability_results.csv        ({len(stab_df)} rows)")

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
files = sorted(os.listdir(OUT))
print(f"\n{'=' * 60}")
print(f"  Exported {len(files)} CSV files to results/tableau_exports/")
print(f"{'=' * 60}")
for fn in files:
    size = os.path.getsize(os.path.join(OUT, fn))
    print(f"    {fn:<35s}  {size:>6,} bytes")

print("\nDone. Import these CSVs into Tableau for dashboard creation.")
