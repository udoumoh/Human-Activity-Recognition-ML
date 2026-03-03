"""
Unit tests for the PAMAP2 Medallion Pipeline.

Tests use PySpark in local mode so no external cluster is needed.
Run with:
    pytest tests/test_pipeline.py -v

The four tests exercise each Medallion layer in isolation and check
correctness invariants that should always hold regardless of data volume.
"""

import os
import sys
import pytest
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import (
    BRONZE_OUTPUT, SILVER_OUTPUT, GOLD_FEATURES_OUTPUT,
    GOLD_MODEL_OUTPUT, TRAIN_TEST_SPLIT, RANDOM_SEED, META_COLS,
)

# ── PySpark session fixture (shared across all tests) ─────────

@pytest.fixture(scope="module")
def spark():
    """Create a local SparkSession for the test module, then stop it."""
    from pyspark.sql import SparkSession
    session = (
        SparkSession.builder
        .appName("PAMAP2_Pipeline_Tests")
        .master("local[2]")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


# ──────────────────────────────────────────────────────────────
# Test 1: Bronze schema
# ──────────────────────────────────────────────────────────────

def test_bronze_schema(spark):
    """
    Bronze layer must produce exactly 54 columns and include a
    correctly typed subject_id column (Integer / Long).
    """
    from pyspark.sql.types import IntegerType, LongType

    df = spark.read.parquet(BRONZE_OUTPUT)

    # Column count check (52 sensor + timestamp + subject_id + activity_id
    # + session_type = 54 per the schema definition)
    assert len(df.columns) == 54, (
        f"Expected 54 Bronze columns, got {len(df.columns)}: {df.columns}"
    )

    # subject_id must be present and numeric
    assert "subject_id" in df.columns, "subject_id column missing from Bronze"
    subject_dtype = df.schema["subject_id"].dataType
    assert isinstance(subject_dtype, (IntegerType, LongType)), (
        f"subject_id should be Int/Long, got {subject_dtype}"
    )

    # Sanity: at least one row and 9 distinct subjects
    row_count = df.count()
    assert row_count > 0, "Bronze parquet is empty"

    subject_count = df.select("subject_id").distinct().count()
    assert subject_count == 9, (
        f"Expected 9 PAMAP2 subjects, found {subject_count}"
    )


# ──────────────────────────────────────────────────────────────
# Test 2: Silver — no HR nulls after preprocessing
# ──────────────────────────────────────────────────────────────

def test_silver_no_nulls(spark):
    """
    After Silver preprocessing the heart_rate column must have zero nulls.
    This validates the forward-fill → back-fill → broadcast-join imputation.
    """
    from pyspark.sql.functions import col

    df = spark.read.parquet(SILVER_OUTPUT)

    assert "heart_rate" in df.columns, "heart_rate column missing from Silver"

    hr_nulls = df.filter(col("heart_rate").isNull()).count()
    assert hr_nulls == 0, (
        f"Silver heart_rate still has {hr_nulls} null(s) — "
        "interpolation incomplete"
    )


# ──────────────────────────────────────────────────────────────
# Test 3: Gold — feature count
# ──────────────────────────────────────────────────────────────

def test_gold_feature_count(spark):
    """
    Gold feature engineering must produce exactly 172 numeric feature
    columns (excluding META_COLS: subject_id, activity_id, timestamp,
    session_type).
    """
    from pyspark.sql.types import DoubleType

    df = spark.read.parquet(GOLD_FEATURES_OUTPUT)

    feature_cols = [
        c for c in df.columns
        if c not in META_COLS
        and isinstance(df.schema[c].dataType, DoubleType)
    ]

    assert len(feature_cols) == 172, (
        f"Expected 172 Gold features, got {len(feature_cols)}"
    )


# ──────────────────────────────────────────────────────────────
# Test 4: Model accuracy floor
# ──────────────────────────────────────────────────────────────

def test_model_accuracy_floor(spark):
    """
    The saved best MLP model must achieve at least 85% accuracy on the
    held-out test split (same split seed used during training).

    This guards against a degraded or incorrectly saved model.
    """
    from pyspark.sql.types import DoubleType
    from pyspark.sql.functions import col, isnan, when
    from pyspark.ml import PipelineModel
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    df = spark.read.parquet(GOLD_FEATURES_OUTPUT)

    feature_cols = sorted([
        c for c in df.columns
        if c not in META_COLS
        and isinstance(df.schema[c].dataType, DoubleType)
    ])

    for c in feature_cols:
        df = df.withColumn(c, when(isnan(col(c)), 0.0).otherwise(col(c)))
    df = df.na.drop(subset=feature_cols)

    _, test_df = df.randomSplit(TRAIN_TEST_SPLIT, seed=RANDOM_SEED)

    model = PipelineModel.load(GOLD_MODEL_OUTPUT)
    preds = model.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", metricName="accuracy"
    )
    accuracy = evaluator.evaluate(preds)

    assert accuracy >= 0.85, (
        f"MLP test accuracy {accuracy:.4f} is below the 0.85 floor — "
        "check that the correct model was saved"
    )
