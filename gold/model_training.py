"""
Gold Layer — MLlib Model Training (Distributed).

This script trains four classification models on the Gold feature dataset
using PySpark MLlib's Pipeline and CrossValidator APIs. All training,
cross-validation, and hyperparameter tuning is distributed across
Spark executors.

Models Trained
--------------
1. Logistic Regression    (multinomial, L2, regParam grid)
2. Random Forest          (numTrees grid, maxDepth=5)
3. Multilayer Perceptron  (64-unit hidden layer, maxIter grid)
4. Linear SVM (OneVsRest) (regParam grid)

Each model is wrapped in a Pipeline:
    StringIndexer -> VectorAssembler -> StandardScaler -> Classifier

CrossValidator performs 3-fold CV with F1 evaluation to select the
best hyperparameters. The best overall model is saved as a
PipelineModel for downstream evaluation.

Distributed Training Notes
--------------------------
- CrossValidator distributes fold evaluation: each fold trains on a
  different data split in parallel (controlled by `parallelism`).
- VectorAssembler and StandardScaler run as distributed transformations
  inside the Pipeline, not as driver-side operations.
- The Pipeline API ensures that preprocessing (indexing, scaling) is
  fitted on training data only and applied consistently to test data,
  preventing data leakage across CV folds.

Usage
-----
    python -m gold.model_training

Input
-----
    data/gold/pamap2_features.parquet

Output
------
    data/gold/model_results.json
    data/gold/best_model/   (saved PipelineModel)
"""

import os
import sys
import time
import json
import logging

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, isnan, when

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    MultilayerPerceptronClassifier,
    LinearSVC,
    OneVsRest,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ── Project imports ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    GOLD_FEATURES_OUTPUT, GOLD_MODEL_OUTPUT, GOLD_RESULTS_OUTPUT,
    SPARK_DRIVER_MEMORY, SPARK_SHUFFLE_PARTITIONS,
    TRAIN_TEST_SPLIT, RANDOM_SEED, CV_FOLDS, META_COLS,
)

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GOLD-TRAIN] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run():
    """Execute the Gold model training pipeline."""

    log.info("=" * 60)
    log.info("GOLD LAYER — MLlib Model Training")
    log.info("=" * 60)

    # ── 1. Initialise Spark ──────────────────────────────────
    spark = (
        SparkSession.builder
        .appName("PAMAP2_Gold_ModelTraining")
        .master("local[4]")
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.python.worker.reuse", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    log.info(f"Spark version: {spark.version}")

    # ── 1b. Configure checkpoint directory ───────────────────
    # Setting a checkpoint directory enables Spark MLlib to
    # periodically write intermediate model state during iterative
    # training (e.g., MLP gradient descent, tree ensemble building).
    # This prevents job failure on long runs and provides recovery
    # points if a task is lost due to executor failure.
    checkpoint_dir = os.path.join(
        os.path.dirname(GOLD_FEATURES_OUTPUT), "checkpoints"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    spark.sparkContext.setCheckpointDir(checkpoint_dir)
    log.info(f"Checkpoint directory: {checkpoint_dir}")

    # ── 2. Load Gold features ────────────────────────────────
    log.info(f"Reading Gold features from: {GOLD_FEATURES_OUTPUT}")
    df = spark.read.parquet(GOLD_FEATURES_OUTPUT)
    log.info(f"Loaded {df.count():,} rows x {len(df.columns)} columns")

    # Identify feature columns
    feature_cols = sorted([
        c for c in df.columns
        if c not in META_COLS
        and isinstance(df.schema[c].dataType, DoubleType)
    ])
    log.info(f"Feature columns: {len(feature_cols)}")

    # Replace NaN with 0 (stddev of constant windows = NaN)
    for c in feature_cols:
        df = df.withColumn(c, when(isnan(col(c)), 0.0).otherwise(col(c)))
    df_clean = df.na.drop(subset=feature_cols)
    log.info(f"After NaN-fix: {df_clean.count():,} rows")

    # ── 3. Train/test split ──────────────────────────────────
    train_df, test_df = df_clean.randomSplit(TRAIN_TEST_SPLIT, seed=RANDOM_SEED)
    train_df.cache()
    test_df.cache()
    log.info(f"Train: {train_df.count():,}  Test: {test_df.count():,}")

    # ── 4. Shared pipeline stages ────────────────────────────
    label_indexer = StringIndexer(
        inputCol="activity_id", outputCol="label"
    ).setHandleInvalid("keep")

    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="features_raw",
        handleInvalid="keep",
    )

    scaler = StandardScaler(
        inputCol="features_raw", outputCol="features",
        withMean=True, withStd=True,
    )

    eval_acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    eval_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    results = []

    def evaluate_model(name, cv_model, test_data):
        """Evaluate a trained CV model and record metrics."""
        t0 = time.time()
        preds = cv_model.transform(test_data)
        acc = eval_acc.evaluate(preds)
        f1 = eval_f1.evaluate(preds)
        elapsed = time.time() - t0

        results.append({
            "model": name,
            "accuracy": round(acc, 4),
            "f1_weighted": round(f1, 4),
            "eval_time_s": round(elapsed, 1),
        })

        log.info(f"{'=' * 56}")
        log.info(f"  {name}")
        log.info(f"  Accuracy    : {acc:.4f}")
        log.info(f"  Weighted F1 : {f1:.4f}")
        log.info(f"  Eval time   : {elapsed:.1f}s")
        log.info(f"{'=' * 56}")

    # ── 5a. Logistic Regression ──────────────────────────────
    log.info("Training: Logistic Regression")
    lr = LogisticRegression(
        featuresCol="features", labelCol="label",
        family="multinomial", maxIter=100, elasticNetParam=0.0,
    )
    lr_pipe = Pipeline(stages=[label_indexer, assembler, scaler, lr])
    lr_grid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1]).build()
    lr_cv = CrossValidator(
        estimator=lr_pipe, estimatorParamMaps=lr_grid,
        evaluator=eval_f1, numFolds=CV_FOLDS, parallelism=1,
        seed=RANDOM_SEED,
    )
    t0 = time.time()
    lr_model = lr_cv.fit(train_df)
    log.info(f"LR training: {time.time() - t0:.1f}s")
    evaluate_model("Logistic Regression", lr_model, test_df)

    # ── 5b. Random Forest ────────────────────────────────────
    log.info("Training: Random Forest")
    rf = RandomForestClassifier(
        featuresCol="features", labelCol="label", seed=RANDOM_SEED,
    )
    rf_pipe = Pipeline(stages=[label_indexer, assembler, scaler, rf])
    rf_grid = (ParamGridBuilder()
               .addGrid(rf.numTrees, [20, 50])
               .addGrid(rf.maxDepth, [5])
               .build())
    rf_cv = CrossValidator(
        estimator=rf_pipe, estimatorParamMaps=rf_grid,
        evaluator=eval_f1, numFolds=CV_FOLDS, parallelism=1,
        seed=RANDOM_SEED,
    )
    t0 = time.time()
    rf_model = rf_cv.fit(train_df)
    log.info(f"RF training: {time.time() - t0:.1f}s")
    evaluate_model("Random Forest", rf_model, test_df)

    # ── 5c. Multilayer Perceptron ────────────────────────────
    log.info("Training: Multilayer Perceptron")
    num_features = len(feature_cols)
    num_classes = train_df.select("activity_id").distinct().count()

    mlp = MultilayerPerceptronClassifier(
        featuresCol="features", labelCol="label",
        layers=[num_features, 64, num_classes],
        blockSize=128, seed=RANDOM_SEED,
    )
    mlp_pipe = Pipeline(stages=[label_indexer, assembler, scaler, mlp])
    mlp_grid = ParamGridBuilder().addGrid(mlp.maxIter, [50, 100]).build()
    mlp_cv = CrossValidator(
        estimator=mlp_pipe, estimatorParamMaps=mlp_grid,
        evaluator=eval_f1, numFolds=CV_FOLDS, parallelism=1,
        seed=RANDOM_SEED,
    )
    t0 = time.time()
    mlp_model = mlp_cv.fit(train_df)
    log.info(f"MLP training: {time.time() - t0:.1f}s")
    evaluate_model("Multilayer Perceptron", mlp_model, test_df)

    # ── 5d. Linear SVM (OneVsRest) ───────────────────────────
    log.info("Training: Linear SVM (OneVsRest)")
    lsvc = LinearSVC(
        featuresCol="features", labelCol="label", maxIter=50,
    )
    ovr = OneVsRest(
        classifier=lsvc, featuresCol="features", labelCol="label",
    )
    svm_pipe = Pipeline(stages=[label_indexer, assembler, scaler, ovr])
    svm_grid = ParamGridBuilder().addGrid(lsvc.regParam, [0.01, 0.1]).build()
    svm_cv = CrossValidator(
        estimator=svm_pipe, estimatorParamMaps=svm_grid,
        evaluator=eval_f1, numFolds=CV_FOLDS, parallelism=1,
        seed=RANDOM_SEED,
    )
    t0 = time.time()
    svm_model = svm_cv.fit(train_df)
    log.info(f"SVM training: {time.time() - t0:.1f}s")
    evaluate_model("Linear SVM (OVR)", svm_model, test_df)

    # ── 6. Summary ───────────────────────────────────────────
    log.info("=" * 64)
    log.info("  MODEL COMPARISON")
    log.info("=" * 64)
    for r in sorted(results, key=lambda x: x["f1_weighted"], reverse=True):
        log.info(f"  {r['model']:25s}  Acc={r['accuracy']:.4f}  F1={r['f1_weighted']:.4f}")

    best = max(results, key=lambda r: r["f1_weighted"])
    log.info(f"Best: {best['model']} (F1={best['f1_weighted']})")

    # ── 7. Save results ──────────────────────────────────────
    os.makedirs(os.path.dirname(GOLD_RESULTS_OUTPUT), exist_ok=True)

    with open(GOLD_RESULTS_OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {GOLD_RESULTS_OUTPUT}")

    # Save best model
    best_models = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
        "Multilayer Perceptron": mlp_model,
        "Linear SVM (OVR)": svm_model,
    }
    best_models[best["model"]].bestModel.write().overwrite().save(GOLD_MODEL_OUTPUT)
    log.info(f"Best model saved to {GOLD_MODEL_OUTPUT}")

    train_df.unpersist()
    test_df.unpersist()
    spark.stop()
    log.info("Gold model training complete.")


if __name__ == "__main__":
    run()
