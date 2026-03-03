"""
Gold Layer — Model Evaluation (Distributed + Local Visualisation).

Evaluates the best trained model (MLP) on the PAMAP2 test set, generating:

1. Overall metrics (accuracy, F1, precision, recall)
2. Confusion matrix heatmaps (raw counts + normalised)
3. Per-class classification report (precision, recall, F1)
4. Per-class F1 bar chart
5. Feature importance via Random Forest surrogate model
6. Confusion matrix in long format for Tableau

Usage
-----
    python -m gold.evaluation

Input
-----
    data/gold/pamap2_features.parquet
    data/gold/best_model/

Output
------
    results/confusion_matrix.png
    results/confusion_matrix.csv    (long format for Tableau)
    results/per_class_f1.png
    results/per_class_metrics.csv
    results/feature_importance.png
    results/feature_importance.csv
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, isnan, when
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    GOLD_FEATURES_OUTPUT, GOLD_MODEL_OUTPUT,
    SPARK_DRIVER_MEMORY, RESULTS_DIR,
    TRAIN_TEST_SPLIT, RANDOM_SEED, META_COLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GOLD-EVAL] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ACTIVITY_NAMES = {
    1: "Lying", 2: "Sitting", 3: "Standing", 4: "Walking",
    5: "Running", 6: "Cycling", 7: "Nordic Walking",
    9: "Watching TV", 10: "Computer Work", 11: "Car Driving",
    12: "Ascending Stairs", 13: "Descending Stairs",
    16: "Vacuum Cleaning", 17: "Ironing", 18: "Folding Laundry",
    19: "House Cleaning", 20: "Playing Soccer", 24: "Rope Jumping",
}

plt.rcParams.update({"figure.dpi": 120, "font.size": 9})


def run():
    """Execute the Gold evaluation pipeline."""

    log.info("=" * 60)
    log.info("GOLD LAYER — Model Evaluation")
    log.info("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("PAMAP2_Gold_Evaluation")
        .master("local[2]")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    log.info(f"Loading features from: {GOLD_FEATURES_OUTPUT}")
    df = spark.read.parquet(GOLD_FEATURES_OUTPUT)

    feature_cols = sorted([
        c for c in df.columns
        if c not in META_COLS
        and isinstance(df.schema[c].dataType, DoubleType)
    ])

    for c in feature_cols:
        df = df.withColumn(c, when(isnan(col(c)), 0.0).otherwise(col(c)))
    df_clean = df.na.drop(subset=feature_cols)

    train_df, test_df = df_clean.randomSplit(TRAIN_TEST_SPLIT, seed=RANDOM_SEED)
    train_df.cache()
    test_df.cache()
    log.info(f"Train: {train_df.count():,}  Test: {test_df.count():,}")

    log.info(f"Loading model from: {GOLD_MODEL_OUTPUT}")
    model = PipelineModel.load(GOLD_MODEL_OUTPUT)
    log.info(f"Pipeline stages: {[type(s).__name__ for s in model.stages]}")

    si_model = model.stages[0]
    idx_to_activity = {i: int(lbl) for i, lbl in enumerate(si_model.labels)}
    idx_to_name = {i: ACTIVITY_NAMES.get(aid, str(aid))
                   for i, aid in idx_to_activity.items()}
    label_names = [idx_to_name[i] for i in range(len(idx_to_name))]

    predictions = model.transform(test_df)

    evaluators = {
        "accuracy": MulticlassClassificationEvaluator(
            labelCol="label", metricName="accuracy"),
        "f1_weighted": MulticlassClassificationEvaluator(
            labelCol="label", metricName="f1"),
        "precision_weighted": MulticlassClassificationEvaluator(
            labelCol="label", metricName="weightedPrecision"),
        "recall_weighted": MulticlassClassificationEvaluator(
            labelCol="label", metricName="weightedRecall"),
    }

    log.info("Overall Test Metrics:")
    for name, ev in evaluators.items():
        val = ev.evaluate(predictions)
        log.info(f"  {name:25s}: {val:.4f}")

    # Confusion matrix computed locally — test set (1,031 rows) fits in driver memory
    pred_pd = predictions.select("label", "prediction").toPandas()
    y_true = pred_pd["label"].astype(int).values
    y_pred = pred_pd["prediction"].astype(int).values

    present_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    present_names = [idx_to_name[i] for i in present_labels]

    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=present_names, yticklabels=present_names,
                ax=axes[0], cbar_kws={"shrink": 0.8})
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=present_names, yticklabels=present_names,
                ax=axes[1], vmin=0, vmax=1, cbar_kws={"shrink": 0.8})
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Normalised Confusion Matrix (recall)")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {cm_path}")

    report_dict = classification_report(
        y_true, y_pred, labels=present_labels,
        target_names=present_names, output_dict=True,
    )

    rows = []
    for name in present_names:
        r = report_dict[name]
        rows.append({
            "activity": name,
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1_score": round(r["f1-score"], 4),
            "support": int(r["support"]),
        })

    report_df = pd.DataFrame(rows).sort_values("f1_score", ascending=True)
    report_df.to_csv(
        os.path.join(RESULTS_DIR, "per_class_metrics.csv"), index=False,
    )
    log.info("Saved per_class_metrics.csv")

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#e74c3c" if f < 0.7 else "#f39c12" if f < 0.85
              else "#2ecc71" for f in report_df["f1_score"]]
    bars = ax.barh(report_df["activity"], report_df["f1_score"], color=colors)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Activity F1 Score (MLP)")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.85, color="gray", linestyle="--", linewidth=0.8,
               label="0.85 threshold")
    for bar, val in zip(bars, report_df["f1_score"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.legend(loc="lower right")
    plt.tight_layout()
    f1_path = os.path.join(RESULTS_DIR, "per_class_f1.png")
    plt.savefig(f1_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {f1_path}")

    log.info("Training RF surrogate for feature importance...")
    label_idx = StringIndexer(
        inputCol="activity_id", outputCol="label"
    ).setHandleInvalid("keep")
    asm = VectorAssembler(
        inputCols=feature_cols, outputCol="features_raw",
        handleInvalid="keep",
    )
    scl = StandardScaler(
        inputCol="features_raw", outputCol="features",
        withMean=True, withStd=True,
    )
    rf = RandomForestClassifier(
        featuresCol="features", labelCol="label",
        numTrees=50, maxDepth=5, seed=RANDOM_SEED,
    )
    rf_pipe = Pipeline(stages=[label_idx, asm, scl, rf])
    rf_model = rf_pipe.fit(train_df)

    importances = rf_model.stages[-1].featureImportances.toArray()
    feat_imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    feat_imp_df.to_csv(
        os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False,
    )
    log.info("Saved feature_importance.csv")

    top20 = feat_imp_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top20["feature"], top20["importance"], color="steelblue")
    ax.set_xlabel("Importance (Gini)")
    ax.set_title("Top 20 Features by Random Forest Importance")
    plt.tight_layout()
    fi_path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {fi_path}")

    cm_rows = []
    for i, true_idx in enumerate(present_labels):
        for j, pred_idx in enumerate(present_labels):
            cm_rows.append({
                "true_label": idx_to_name[true_idx],
                "predicted_label": idx_to_name[pred_idx],
                "count": int(cm[i, j]),
                "normalised": round(float(cm_norm[i, j]), 4),
            })

    pd.DataFrame(cm_rows).to_csv(
        os.path.join(RESULTS_DIR, "confusion_matrix.csv"), index=False,
    )
    log.info("Saved confusion_matrix.csv (long format)")

    train_df.unpersist()
    test_df.unpersist()
    spark.stop()
    log.info("Gold evaluation complete.")


if __name__ == "__main__":
    run()
