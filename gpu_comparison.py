"""
GPU vs CPU vs Distributed ML Comparison — PAMAP2 Human Activity Recognition.

This script provides a three-way performance comparison:

  1. PySpark MLlib MLP     — distributed, runs on the JVM Spark engine
  2. PyTorch MLP (CPU)     — single-node, NumPy/PyTorch on the host CPU
  3. PyTorch MLP (CUDA)    — single-node, same architecture on the NVIDIA GPU

Architecture
------------
The neural-network architecture mirrors the best MLlib config found during
hyperparameter tuning:
    Input (172) → Hidden (64) → Output (18 classes)
    Activation : ReLU (hidden), LogSoftmax (output)
    Loss       : NLLLoss
    Optimiser  : Adam, lr=1e-3, 100 epochs

Metrics Recorded
----------------
- Training wall-clock time (seconds)
- Test accuracy
- Weighted F1 score (via sklearn)
- GPU memory peak allocated (MB, CUDA run only)
- CUDA device name and compute capability

Output
------
    results/gpu_comparison.csv
    results/gpu_comparison.json

Usage
-----
    python gpu_comparison.py
"""

import os
import sys
import time
import json
import csv
import logging

# ── Ensure PySpark workers use the same Python interpreter ────
# On Windows the bare `python` command may resolve to the
# Microsoft Store stub instead of the real interpreter.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import numpy as np
import pandas as pd

# ── Project imports ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config.settings import (
    GOLD_FEATURES_OUTPUT, RESULTS_DIR,
    TRAIN_TEST_SPLIT, RANDOM_SEED, META_COLS,
)

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GPU-CMP] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 1. Shared Spark session builder
# ──────────────────────────────────────────────────────────────

def get_spark():
    """
    Return (or create) a single shared SparkSession.
    Using one session across data loading AND MLlib avoids starting
    a second JVM, which causes native memory exhaustion on Windows.
    JVM flags: G1GC + enlarged code cache prevent heap OOM during MLP training.
    """
    from pyspark.sql import SparkSession
    return (
        SparkSession.builder
        .appName("GPU_Comparison")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.ui.enabled", "false")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


# ──────────────────────────────────────────────────────────────
# 2. PyTorch MLP definition
# ──────────────────────────────────────────────────────────────

def build_mlp(n_features: int, n_classes: int):
    """Return a two-layer MLP: Input→64→n_classes."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Linear(64, n_classes),
        nn.LogSoftmax(dim=1),
    )


def train_pytorch(X_train, y_train, X_test, y_test,
                  n_features, n_classes, device_name: str):
    """
    Train a PyTorch MLP on the given device, return metrics dict.

    Parameters
    ----------
    device_name : "cpu" or "cuda"
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import f1_score, accuracy_score

    device = torch.device(device_name)

    # GPU info (only meaningful for CUDA)
    gpu_name = "N/A"
    compute_cap = "N/A"
    gpu_mem_mb = 0.0

    if device_name == "cuda":
        if not torch.cuda.is_available():
            log.warning("CUDA requested but not available — skipping GPU run.")
            return None
        gpu_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        compute_cap = f"{cap[0]}.{cap[1]}"
        torch.cuda.reset_peak_memory_stats(0)
        log.info(f"  CUDA device : {gpu_name}  (compute {compute_cap})")

    torch.manual_seed(RANDOM_SEED)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test,  dtype=torch.float32)
    y_te = torch.tensor(y_test,  dtype=torch.long)

    dataset = TensorDataset(X_tr, y_tr)
    loader  = DataLoader(dataset, batch_size=128, shuffle=True)

    model = build_mlp(n_features, n_classes).to(device)
    criterion = nn.NLLLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ── Training ────────────────────────────────────────────
    t0 = time.time()
    model.train()
    for epoch in range(100):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
    train_sec = round(time.time() - t0, 2)

    # ── Inference ───────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        X_te_dev = X_te.to(device)
        logits = model(X_te_dev)
        preds = logits.argmax(dim=1).cpu().numpy()

    acc  = round(float(accuracy_score(y_test, preds)), 4)
    f1   = round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4)

    if device_name == "cuda":
        gpu_mem_mb = round(
            torch.cuda.max_memory_allocated(0) / (1024 ** 2), 1
        )

    return {
        "framework":      f"PyTorch ({device_name.upper()})",
        "device":         gpu_name if device_name == "cuda" else "CPU",
        "compute_cap":    compute_cap,
        "train_sec":      train_sec,
        "accuracy":       acc,
        "f1_weighted":    f1,
        "gpu_mem_peak_mb": gpu_mem_mb,
    }


# ──────────────────────────────────────────────────────────────
# 3. PySpark MLlib baseline (reuses shared Spark session)
# ──────────────────────────────────────────────────────────────

def run_mllib_baseline(spark, train_sdf, test_sdf, n_features, n_classes):
    """
    Run PySpark MLlib MLP with architecture [n_features, 64, n_classes]
    using the shared Spark session and pre-split native Spark DataFrames.
    Avoids creating a second JVM or converting Python objects to DataFrames.
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
    from pyspark.ml.classification import MultilayerPerceptronClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from sklearn.metrics import f1_score

    feature_cols = [c for c in train_sdf.columns if c != "activity_id"]

    si  = StringIndexer(inputCol="activity_id", outputCol="label",
                        handleInvalid="keep")
    asm = VectorAssembler(inputCols=feature_cols, outputCol="features_raw",
                          handleInvalid="keep")
    scl = StandardScaler(inputCol="features_raw", outputCol="features",
                         withMean=True, withStd=True)
    mlp = MultilayerPerceptronClassifier(
        featuresCol="features",
        labelCol="label",
        layers=[n_features, 64, n_classes],
        blockSize=128,
        maxIter=50,
        seed=RANDOM_SEED,
    )
    pipe = Pipeline(stages=[si, asm, scl, mlp])

    t0 = time.time()
    model = pipe.fit(train_sdf)
    train_sec = round(time.time() - t0, 2)

    preds_sdf = model.transform(test_sdf)

    ev_acc = MulticlassClassificationEvaluator(
        labelCol="label", metricName="accuracy")
    acc = round(ev_acc.evaluate(preds_sdf), 4)

    pred_pd = preds_sdf.select("label", "prediction").toPandas()
    f1 = round(float(f1_score(
        pred_pd["label"].astype(int),
        pred_pd["prediction"].astype(int),
        average="weighted", zero_division=0,
    )), 4)

    return {
        "framework":       "PySpark MLlib MLP",
        "device":          "Distributed (local[2])",
        "compute_cap":     "N/A",
        "train_sec":       train_sec,
        "accuracy":        acc,
        "f1_weighted":     f1,
        "gpu_mem_peak_mb": 0.0,
    }


# ──────────────────────────────────────────────────────────────
# 4. Main
# ──────────────────────────────────────────────────────────────

def run():
    log.info("=" * 60)
    log.info("GPU vs CPU vs MLlib Comparison")
    log.info("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Single Spark session (reused for data load + MLlib) ─
    from pyspark.sql.types import DoubleType
    from pyspark.sql.functions import col, isnan, when
    from sklearn.preprocessing import LabelEncoder

    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")

    log.info("Loading Gold features …")
    df = spark.read.parquet(GOLD_FEATURES_OUTPUT)
    feature_cols = sorted([
        c for c in df.columns
        if c not in META_COLS
        and isinstance(df.schema[c].dataType, DoubleType)
    ])
    for c in feature_cols:
        df = df.withColumn(c, when(isnan(col(c)), 0.0).otherwise(col(c)))
    df = df.na.drop(subset=feature_cols)

    n_features = len(feature_cols)
    n_classes  = int(df.select("activity_id").distinct().count())

    # Native Spark split — used by MLlib (no Python↔Spark round-trip)
    train_sdf, test_sdf = df.randomSplit(TRAIN_TEST_SPLIT, seed=RANDOM_SEED)

    # NumPy arrays — used by PyTorch
    pd_df = df.select(feature_cols + ["activity_id"]).toPandas()
    X     = pd_df[feature_cols].values.astype(np.float32)
    le    = LabelEncoder()
    y     = le.fit_transform(pd_df["activity_id"].values).astype(np.int64)

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(X))
    split = int(len(idx) * TRAIN_TEST_SPLIT[0])
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    log.info(f"  Train: {len(X_train):,}  Test: {len(X_test):,}  "
             f"Features: {n_features}  Classes: {n_classes}")

    results = []

    # ── Run 1: PyTorch CPU ──────────────────────────────────
    log.info("\n[1/3] PyTorch MLP — CPU")
    cpu_result = train_pytorch(
        X_train, y_train, X_test, y_test, n_features, n_classes, "cpu"
    )
    results.append(cpu_result)
    log.info(f"  time={cpu_result['train_sec']}s  "
             f"acc={cpu_result['accuracy']}  f1={cpu_result['f1_weighted']}")

    # ── Run 2: PyTorch CUDA ─────────────────────────────────
    import torch
    if torch.cuda.is_available():
        log.info("\n[2/3] PyTorch MLP — CUDA GPU")
        gpu_result = train_pytorch(
            X_train, y_train, X_test, y_test, n_features, n_classes, "cuda"
        )
        if gpu_result:
            results.append(gpu_result)
            log.info(f"  time={gpu_result['train_sec']}s  "
                     f"acc={gpu_result['accuracy']}  f1={gpu_result['f1_weighted']}  "
                     f"peak_mem={gpu_result['gpu_mem_peak_mb']} MB")
    else:
        log.warning("[2/3] CUDA not available — skipping GPU run.")

    # ── Run 3: PySpark MLlib ────────────────────────────────
    log.info("\n[3/3] PySpark MLlib MLP — Distributed")
    mllib_result = run_mllib_baseline(
        spark, train_sdf, test_sdf, n_features, n_classes
    )
    results.append(mllib_result)
    log.info(f"  time={mllib_result['train_sec']}s  "
             f"acc={mllib_result['accuracy']}  f1={mllib_result['f1_weighted']}")

    spark.stop()

    # ── Summary table ───────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("COMPARISON SUMMARY")
    log.info("=" * 60)
    header = f"{'Framework':<28} {'Device':<22} {'Time(s)':>8} {'Accuracy':>10} {'F1':>8} {'GPU Mem(MB)':>12}"
    log.info(header)
    log.info("-" * len(header))
    for r in results:
        log.info(
            f"{r['framework']:<28} {r['device']:<22} "
            f"{r['train_sec']:>8.1f} {r['accuracy']:>10.4f} "
            f"{r['f1_weighted']:>8.4f} {r['gpu_mem_peak_mb']:>12.1f}"
        )

    # ── Save results ────────────────────────────────────────
    csv_path  = os.path.join(RESULTS_DIR, "gpu_comparison.csv")
    json_path = os.path.join(RESULTS_DIR, "gpu_comparison.json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    log.info(f"\nSaved {csv_path}")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved {json_path}")

    log.info("GPU comparison complete.")


if __name__ == "__main__":
    run()
