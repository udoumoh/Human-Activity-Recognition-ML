"""
Gold Layer — Tableau Dashboard Export.

This script consolidates all pipeline results into clean, flat CSV
files optimised for Tableau dashboard import. Tableau performs best
with pre-aggregated "Gold" tables rather than raw data, so we use
Spark and pandas to aggregate millions of records into summary
tables that the dashboard can load instantly.

Medallion Architecture Context
------------------------------
This is the final output of the Gold layer — analytics-ready
summary tables for business intelligence consumption. Each CSV
is designed as a single Tableau data source, avoiding the need
for complex joins inside the dashboard.

Output CSVs (results/tableau_exports/)
--------------------------------------
1.  dataset_summary.csv      — Dataset overview metrics
2.  model_comparison.csv     — All 4 Spark model results
3.  spark_vs_sklearn.csv     — PySpark vs scikit-learn comparison
4.  confusion_matrix.csv     — 18x18 matrix in long format
5.  per_class_metrics.csv    — Per-activity precision/recall/F1
6.  feature_importance.csv   — All 172 features ranked
7.  scaling_results.csv      — Strong + weak scaling combined
8.  stability_results.csv    — MLP 5-seed stability runs
9.  gpu_comparison.csv       — MLlib vs PyTorch-CPU vs PyTorch-GPU
10. roc_curves.csv           — One-vs-Rest ROC points for all 18 classes
11. extended_metrics.csv     — 6-metric table (adds Cohen's Kappa, MCC)
12. significance_test.csv    — McNemar's test: MLP vs Logistic Regression

Usage
-----
    python -m gold.tableau_export
"""

import os
import sys
import json
import logging

import pandas as pd

# ── Project imports ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    PROJECT_ROOT, RESULTS_DIR, TABLEAU_EXPORT_DIR,
    GOLD_RESULTS_OUTPUT,
)

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GOLD-EXPORT] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run():
    """Consolidate all results into Tableau-ready CSVs."""

    log.info("=" * 60)
    log.info("GOLD LAYER — Tableau Export")
    log.info("=" * 60)

    os.makedirs(TABLEAU_EXPORT_DIR, exist_ok=True)

    # ── 1. Dataset Summary ───────────────────────────────────
    dataset_summary = pd.DataFrame([
        {"metric": "Raw Data Size (GB)",    "value": 1.61},
        {"metric": "Raw Rows",              "value": 3851586},
        {"metric": "Raw Columns",           "value": 54},
        {"metric": "Subjects",              "value": 9},
        {"metric": "Activities",            "value": 18},
        {"metric": "Sampling Rate (Hz)",    "value": 100},
        {"metric": "Window Duration (s)",   "value": 5.0},
        {"metric": "Windowed Feature Rows", "value": 5447},
        {"metric": "Feature Columns",       "value": 172},
        {"metric": "Min Window Fill",       "value": 0.5},
    ])
    _save(dataset_summary, "dataset_summary.csv", "1/12")

    # ── 2. Model Comparison (all 4 Spark models) ─────────────
    results_path = GOLD_RESULTS_OUTPUT
    # Fall back to legacy location if Gold results don't exist yet
    if not os.path.exists(results_path):
        results_path = os.path.join(PROJECT_ROOT, "data", "model_results.json")

    with open(results_path) as f:
        model_data = json.load(f)

    model_df = pd.DataFrame(model_data)
    model_df["framework"] = "PySpark MLlib"
    model_df = model_df.rename(columns={"model": "model_name"})
    _save(model_df, "model_comparison.csv", "2/12")

    # ── 3. Spark vs Sklearn Comparison ───────────────────────
    sklearn_path = os.path.join(RESULTS_DIR, "sklearn_comparison.csv")
    if os.path.exists(sklearn_path):
        sklearn_df = pd.read_csv(sklearn_path)
        _save(sklearn_df, "spark_vs_sklearn.csv", "3/12")
    else:
        log.warning("  sklearn_comparison.csv not found — skipping")

    # ── 4. Confusion Matrix (long format) ────────────────────
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.csv")
    if os.path.exists(cm_path):
        cm_df = pd.read_csv(cm_path)
        _save(cm_df, "confusion_matrix.csv", "4/12")
    else:
        log.warning("  confusion_matrix.csv not found — skipping")

    # ── 5a. Per-Class Metrics ────────────────────────────────
    pc_path = os.path.join(RESULTS_DIR, "per_class_metrics.csv")
    if os.path.exists(pc_path):
        pc_df = pd.read_csv(pc_path)
        _save(pc_df, "per_class_metrics.csv", "5a/12")
    else:
        log.warning("  per_class_metrics.csv not found — skipping")

    # ── 5b. Feature Importance ───────────────────────────────
    fi_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
    if os.path.exists(fi_path):
        fi_df = pd.read_csv(fi_path)
        _save(fi_df, "feature_importance.csv", "5b/12")
    else:
        log.warning("  feature_importance.csv not found — skipping")

    # ── 6. Scalability Results (strong + weak combined) ──────
    strong_path = os.path.join(PROJECT_ROOT, "data", "scalability_results.csv")
    weak_path = os.path.join(RESULTS_DIR, "weak_scaling.csv")

    if os.path.exists(strong_path):
        strong_df = pd.read_csv(strong_path)
        strong_df["scaling_type"] = "strong"
        strong_df["n_cores"] = strong_df["cores"].str.extract(r"\[(\d+)\]").astype(int)

        baseline_times = (
            strong_df[strong_df["n_cores"] == 1]
            .set_index("fraction")["train_sec"]
        )
        strong_df["speedup"] = strong_df.apply(
            lambda r: round(
                baseline_times.get(r["fraction"], r["train_sec"]) / r["train_sec"], 2
            ) if r["train_sec"] > 0 else 0.0,
            axis=1,
        )

        parts = [strong_df]

        if os.path.exists(weak_path):
            weak_df = pd.read_csv(weak_path)
            weak_df["scaling_type"] = "weak"
            if "n_cores" not in weak_df.columns:
                weak_df["n_cores"] = weak_df["cores"].str.extract(r"\[(\d+)\]").astype(int)
            base_time_weak = weak_df.iloc[0]["train_sec"]
            weak_df["speedup"] = round(base_time_weak / weak_df["train_sec"], 2)
            parts.append(weak_df)

        common_cols = [
            "scaling_type", "cores", "n_cores", "fraction", "rows",
            "train_rows", "train_sec", "accuracy", "f1_weighted", "speedup",
        ]
        scaling_df = pd.concat(
            [p[common_cols] for p in parts], ignore_index=True,
        )
        _save(scaling_df, "scaling_results.csv", "6/12")
    else:
        log.warning("  scalability_results.csv not found — skipping")

    # ── 7. Stability Results ─────────────────────────────────
    stab_path = os.path.join(RESULTS_DIR, "stability_results.csv")
    if os.path.exists(stab_path):
        stab_df = pd.read_csv(stab_path)
        _save(stab_df, "stability_results.csv", "7/12")
    else:
        log.warning("  stability_results.csv not found — skipping")

    # ── 8. GPU Comparison ────────────────────────────────────
    gpu_path = os.path.join(RESULTS_DIR, "gpu_comparison.csv")
    if os.path.exists(gpu_path):
        gpu_df = pd.read_csv(gpu_path)
        _save(gpu_df, "gpu_comparison.csv", "8/12")
    else:
        log.warning("  gpu_comparison.csv not found — run gpu_comparison.py first")

    # ── 9. ROC Curves ────────────────────────────────────────
    roc_path = os.path.join(RESULTS_DIR, "roc_curves.csv")
    if os.path.exists(roc_path):
        roc_df = pd.read_csv(roc_path)
        _save(roc_df, "roc_curves.csv", "9/12")
    else:
        log.warning("  roc_curves.csv not found — run gold/extended_evaluation.py first")

    # ── 10. Extended Metrics (6-metric table) ────────────────
    ext_path = os.path.join(RESULTS_DIR, "extended_metrics.csv")
    if os.path.exists(ext_path):
        ext_df = pd.read_csv(ext_path)
        _save(ext_df, "extended_metrics.csv", "10/12")
    else:
        log.warning("  extended_metrics.csv not found — run gold/extended_evaluation.py first")

    # ── 11. Bootstrap CI ─────────────────────────────────────
    boot_path = os.path.join(RESULTS_DIR, "bootstrap_ci.csv")
    if os.path.exists(boot_path):
        boot_df = pd.read_csv(boot_path)
        _save(boot_df, "bootstrap_ci.csv", "11/12")
    else:
        log.warning("  bootstrap_ci.csv not found — run gold/extended_evaluation.py first")

    # ── 12. McNemar Significance Test ────────────────────────
    sig_path = os.path.join(RESULTS_DIR, "significance_test.csv")
    if os.path.exists(sig_path):
        sig_df = pd.read_csv(sig_path)
        _save(sig_df, "significance_test.csv", "12/12")
    else:
        log.warning("  significance_test.csv not found — run gold/extended_evaluation.py first")

    # ── Summary ──────────────────────────────────────────────
    files = sorted(os.listdir(TABLEAU_EXPORT_DIR))
    log.info(f"Exported {len(files)} CSV files to {TABLEAU_EXPORT_DIR}")
    for fn in files:
        size = os.path.getsize(os.path.join(TABLEAU_EXPORT_DIR, fn))
        log.info(f"  {fn:<35s}  {size:>6,} bytes")

    log.info("Gold Tableau export complete.")


def _save(df: pd.DataFrame, filename: str, label: str):
    """Save a DataFrame to the Tableau export directory."""
    path = os.path.join(TABLEAU_EXPORT_DIR, filename)
    df.to_csv(path, index=False)
    log.info(f"  [{label}] {filename:<35s} ({len(df)} rows)")


if __name__ == "__main__":
    run()
