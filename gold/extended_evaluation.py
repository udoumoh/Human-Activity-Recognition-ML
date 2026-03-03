"""
Gold Layer — Extended Model Evaluation.

Supplements gold/evaluation.py with additional statistical rigour:

Metrics added
-------------
5. Cohen's Kappa              (sklearn.metrics.cohen_kappa_score)
6. Matthews Correlation Coeff (sklearn.metrics.matthews_corrcoef)

Curve analysis
--------------
- One-vs-Rest macro-averaged ROC + AUC for all 18 activities
- Macro-averaged Precision-Recall curve

Statistical tests
-----------------
- Bootstrap CI (1 000 resamples, 95%) for accuracy and weighted F1
- McNemar's test: pairwise comparison of MLP vs Logistic Regression

Output CSVs (results/)
-----------------------
    extended_metrics.csv    — 6 metrics for all available models
    roc_curves.csv          — (fpr, tpr, threshold, auc) per class
    bootstrap_ci.csv        — 95% CI bounds for accuracy & F1
    significance_test.csv   — McNemar p-value, statistic, decision

Usage
-----
    python -m gold.extended_evaluation
"""

import os
import sys
import logging
import time

# ── Ensure PySpark workers use the same Python interpreter ────
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    cohen_kappa_score,
    matthews_corrcoef,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, accuracy_score,
)
from sklearn.preprocessing import label_binarize
from scipy.stats import chi2_contingency

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, isnan, when
from pyspark.ml import PipelineModel, Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    MultilayerPerceptronClassifier,
    LogisticRegression,
)

# ── Project imports ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    GOLD_FEATURES_OUTPUT, GOLD_MODEL_OUTPUT,
    SPARK_DRIVER_MEMORY, RESULTS_DIR,
    TRAIN_TEST_SPLIT, RANDOM_SEED, META_COLS,
)

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EXT-EVAL] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Activity name map ─────────────────────────────────────────
ACTIVITY_NAMES = {
    1: "Lying", 2: "Sitting", 3: "Standing", 4: "Walking",
    5: "Running", 6: "Cycling", 7: "Nordic Walking",
    9: "Watching TV", 10: "Computer Work", 11: "Car Driving",
    12: "Ascending Stairs", 13: "Descending Stairs",
    16: "Vacuum Cleaning", 17: "Ironing", 18: "Folding Laundry",
    19: "House Cleaning", 20: "Playing Soccer", 24: "Rope Jumping",
}


# ──────────────────────────────────────────────────────────────
# Helper: extended scalar metrics
# ──────────────────────────────────────────────────────────────

def extended_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     model_name: str) -> dict:
    """Return 6-metric dict (accuracy, F1, precision, recall, kappa, MCC)."""
    return {
        "model":              model_name,
        "accuracy":           round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_weighted":        round(float(f1_score(y_true, y_pred,
                                    average="weighted", zero_division=0)), 4),
        "precision_weighted": round(float(f1_score(y_true, y_pred,
                                    average="weighted", zero_division=0)), 4),
        "recall_weighted":    round(float(f1_score(y_true, y_pred,
                                    average="weighted", zero_division=0)), 4),
        "cohen_kappa":        round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "mcc":                round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }


# ──────────────────────────────────────────────────────────────
# Helper: ROC curves (One-vs-Rest)
# ──────────────────────────────────────────────────────────────

def compute_roc_curves(y_true: np.ndarray, y_prob: np.ndarray,
                       class_labels: list) -> pd.DataFrame:
    """
    Compute per-class and macro-averaged ROC curves.

    Parameters
    ----------
    y_true       : 1-D integer class indices (0 … n_classes-1)
    y_prob       : 2-D float probabilities (n_samples × n_classes)
    class_labels : list of human-readable class names (length n_classes)

    Returns
    -------
    DataFrame with columns: class, fpr, tpr, threshold, auc
    """
    n_classes = y_prob.shape[1]
    classes   = np.arange(n_classes)
    y_bin     = label_binarize(y_true, classes=classes)

    rows = []
    fprs_all, tprs_all, auc_vals = [], [], []

    for i, label in enumerate(class_labels):
        if i >= n_classes:
            continue
        fpr, tpr, thresh = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_vals.append(roc_auc)
        fprs_all.append(fpr)
        tprs_all.append(tpr)
        for j in range(len(fpr)):
            rows.append({
                "class":     label,
                "fpr":       round(float(fpr[j]), 5),
                "tpr":       round(float(tpr[j]), 5),
                "threshold": round(float(thresh[j]) if j < len(thresh) else 1.0, 5),
                "auc":       round(float(roc_auc), 4),
            })

    # Macro average — interpolated on common FPR grid
    mean_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(mean_fpr)
    for fpr, tpr in zip(fprs_all, tprs_all):
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr /= len(fprs_all)
    macro_auc = auc(mean_fpr, mean_tpr)

    for j in range(len(mean_fpr)):
        rows.append({
            "class":     "MACRO_AVG",
            "fpr":       round(float(mean_fpr[j]), 5),
            "tpr":       round(float(mean_tpr[j]), 5),
            "threshold": float("nan"),
            "auc":       round(float(macro_auc), 4),
        })

    log.info(f"  Macro-averaged AUC: {macro_auc:.4f}")
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Helper: Bootstrap CI
# ──────────────────────────────────────────────────────────────

def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Non-parametric bootstrap confidence intervals (1 000 resamples).
    Returns a DataFrame with 95% CI for accuracy and weighted F1.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc_boot, f1_boot = [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        acc_boot.append(accuracy_score(yt, yp))
        f1_boot.append(f1_score(yt, yp, average="weighted", zero_division=0))

    lo, hi = alpha / 2, 1 - alpha / 2

    rows = [
        {
            "metric":  "accuracy",
            "observed": round(float(accuracy_score(y_true, y_pred)), 4),
            "ci_lower": round(float(np.quantile(acc_boot, lo)), 4),
            "ci_upper": round(float(np.quantile(acc_boot, hi)), 4),
            "n_boot":   n_boot,
        },
        {
            "metric":  "f1_weighted",
            "observed": round(float(f1_score(y_true, y_pred,
                               average="weighted", zero_division=0)), 4),
            "ci_lower": round(float(np.quantile(f1_boot, lo)), 4),
            "ci_upper": round(float(np.quantile(f1_boot, hi)), 4),
            "n_boot":   n_boot,
        },
    ]
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Helper: McNemar's test
# ──────────────────────────────────────────────────────────────

def mcnemar_test(y_true: np.ndarray,
                 y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray,
                 name_a: str = "MLP",
                 name_b: str = "Logistic Regression") -> pd.DataFrame:
    """
    McNemar's test for pairwise significance between two classifiers.

    Contingency table:
        b00 = both correct
        b01 = A correct, B wrong
        b10 = A wrong,   B correct
        b11 = both wrong

    McNemar statistic = (b01 - b10)^2 / (b01 + b10)   (chi-squared, df=1)
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b00 = int(np.sum( correct_a &  correct_b))
    b01 = int(np.sum( correct_a & ~correct_b))
    b10 = int(np.sum(~correct_a &  correct_b))
    b11 = int(np.sum(~correct_a & ~correct_b))

    table = np.array([[b00, b01], [b10, b11]])

    # Use chi2 with Yates continuity correction (standard for McNemar)
    if (b01 + b10) == 0:
        statistic, p_value = 0.0, 1.0
    else:
        statistic = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
        from scipy.stats import chi2
        p_value = float(chi2.sf(statistic, df=1))

    decision = "significant" if p_value < 0.05 else "not significant"
    log.info(f"  McNemar ({name_a} vs {name_b}): "
             f"stat={statistic:.4f}  p={p_value:.4f}  → {decision}")

    return pd.DataFrame([{
        "model_a":   name_a,
        "model_b":   name_b,
        "b00":       b00,
        "b01":       b01,
        "b10":       b10,
        "b11":       b11,
        "statistic": round(statistic, 4),
        "p_value":   round(p_value, 6),
        "significant_at_0.05": decision,
    }])


# ──────────────────────────────────────────────────────────────
# ROC plot helper
# ──────────────────────────────────────────────────────────────

def plot_roc(roc_df: pd.DataFrame, save_path: str):
    """Plot per-class + macro ROC curves."""
    classes = [c for c in roc_df["class"].unique() if c != "MACRO_AVG"]
    macro   = roc_df[roc_df["class"] == "MACRO_AVG"]

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.cm.get_cmap("tab20", len(classes))

    for i, cls in enumerate(classes):
        sub = roc_df[roc_df["class"] == cls]
        roc_auc = sub["auc"].iloc[0]
        ax.plot(sub["fpr"], sub["tpr"], lw=0.8, alpha=0.5,
                color=cmap(i), label=f"{cls} (AUC={roc_auc:.2f})")

    macro_auc = macro["auc"].iloc[0]
    ax.plot(macro["fpr"], macro["tpr"], lw=2.5, color="black",
            label=f"Macro avg (AUC={macro_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — One-vs-Rest (MLP, 18 activities)")
    ax.legend(loc="lower right", fontsize=6, ncol=2)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved ROC plot: {save_path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def run():
    log.info("=" * 60)
    log.info("GOLD LAYER — Extended Evaluation")
    log.info("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Spark session ─────────────────────────────────────
    spark = (
        SparkSession.builder
        .appName("PAMAP2_Extended_Evaluation")
        .master("local[2]")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.pyspark.python", sys.executable)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # ── 2. Load data ─────────────────────────────────────────
    log.info(f"Loading features from: {GOLD_FEATURES_OUTPUT}")
    df = spark.read.parquet(GOLD_FEATURES_OUTPUT)

    feature_cols = sorted([
        c for c in df.columns
        if c not in META_COLS
        and isinstance(df.schema[c].dataType, DoubleType)
    ])

    for c in feature_cols:
        df = df.withColumn(c, when(isnan(col(c)), 0.0).otherwise(col(c)))
    df = df.na.drop(subset=feature_cols)

    train_df, test_df = df.randomSplit(TRAIN_TEST_SPLIT, seed=RANDOM_SEED)
    train_df.cache()
    test_df.cache()
    log.info(f"  Train: {train_df.count():,}  Test: {test_df.count():,}")

    n_features = len(feature_cols)
    n_classes = train_df.select("activity_id").distinct().count()

    # ── 3. Load best MLP model ───────────────────────────────
    log.info(f"Loading MLP model from: {GOLD_MODEL_OUTPUT}")
    mlp_model = PipelineModel.load(GOLD_MODEL_OUTPUT)

    si_model     = mlp_model.stages[0]
    idx_to_act   = {i: int(lbl) for i, lbl in enumerate(si_model.labels)}
    idx_to_name  = {i: ACTIVITY_NAMES.get(a, str(a))
                    for i, a in idx_to_act.items()}
    class_labels = [idx_to_name[i] for i in range(len(idx_to_name))]

    mlp_preds = mlp_model.transform(test_df)
    mlp_pd    = mlp_preds.select("label", "prediction", "probability").toPandas()

    y_true     = mlp_pd["label"].astype(int).values
    y_pred_mlp = mlp_pd["prediction"].astype(int).values

    # Probability matrix for ROC (convert DenseVector to numpy)
    y_prob_mlp = np.vstack(
        mlp_pd["probability"].apply(lambda v: np.array(v.toArray())).values
    )

    # ── 4. Train Logistic Regression for McNemar comparison ─
    log.info("Training Logistic Regression for McNemar comparison …")
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
    lr = LogisticRegression(
        featuresCol="features", labelCol="label",
        maxIter=100, regParam=0.01, elasticNetParam=0.0,
        family="multinomial",
    )
    lr_pipe  = Pipeline(stages=[label_idx, asm, scl, lr])
    lr_model = lr_pipe.fit(train_df)
    lr_preds = lr_model.transform(test_df)
    lr_pd    = lr_preds.select("label", "prediction").toPandas()
    y_pred_lr = lr_pd["prediction"].astype(int).values

    # ── 5. Extended metrics ──────────────────────────────────
    log.info("Computing extended metrics …")
    metric_rows = [
        extended_metrics(y_true, y_pred_mlp, "MLP"),
        extended_metrics(y_true, y_pred_lr,  "Logistic Regression"),
    ]
    ext_df = pd.DataFrame(metric_rows)
    log.info("\n" + ext_df.to_string(index=False))

    ext_csv = os.path.join(RESULTS_DIR, "extended_metrics.csv")
    ext_df.to_csv(ext_csv, index=False)
    log.info(f"  Saved {ext_csv}")

    # ── 6. ROC curves ────────────────────────────────────────
    log.info("Computing ROC curves …")
    roc_df  = compute_roc_curves(y_true, y_prob_mlp, class_labels)
    roc_csv = os.path.join(RESULTS_DIR, "roc_curves.csv")
    roc_df.to_csv(roc_csv, index=False)
    log.info(f"  Saved {roc_csv}  ({len(roc_df)} rows)")

    roc_png = os.path.join(RESULTS_DIR, "roc_curves.png")
    plot_roc(roc_df, roc_png)

    # ── 7. Bootstrap CI ──────────────────────────────────────
    log.info("Bootstrap CI (1 000 resamples) …")
    boot_df  = bootstrap_ci(y_true, y_pred_mlp, n_boot=1000)
    boot_csv = os.path.join(RESULTS_DIR, "bootstrap_ci.csv")
    boot_df.to_csv(boot_csv, index=False)
    log.info("\n" + boot_df.to_string(index=False))
    log.info(f"  Saved {boot_csv}")

    # ── 8. McNemar's test ────────────────────────────────────
    log.info("McNemar's significance test …")
    mcnemar_df  = mcnemar_test(y_true, y_pred_mlp, y_pred_lr,
                               "MLP", "Logistic Regression")
    sig_csv = os.path.join(RESULTS_DIR, "significance_test.csv")
    mcnemar_df.to_csv(sig_csv, index=False)
    log.info(f"  Saved {sig_csv}")

    # ── Cleanup ──────────────────────────────────────────────
    train_df.unpersist()
    test_df.unpersist()
    spark.stop()
    log.info("Extended evaluation complete.")


if __name__ == "__main__":
    run()
