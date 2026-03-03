"""
Silver Layer — Data Preprocessing (Distributed).

Value-level transformations on structurally cleaned data from cleaning.py.

Operations
----------
1. Heart-rate interpolation: HR is sampled at ~9 Hz while IMUs run at 100 Hz,
   so ~91% of HR values are null by design. Bounded forward-fill (15 rows),
   back-fill, then per-subject mean imputation via broadcast join.

2. Min-Max normalisation: all sensor columns scaled to [0, 1] using global
   min/max computed in a single distributed aggregation pass.

Usage
-----
    python -m silver.preprocessing

Input
-----
    data/silver/pamap2_cleaned_intermediate.parquet

Output
------
    data/silver/pamap2_clean.parquet
"""

import os
import sys
import logging

from pyspark.sql import SparkSession, Window
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import (
    col, lit, isnan, when, count,
    last, first,
    mean as F_mean,
    min as spark_min,
    max as spark_max,
    broadcast,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    SILVER_OUTPUT, SPARK_DRIVER_MEMORY, SPARK_SHUFFLE_PARTITIONS,
    META_COLS, HR_FILL_WINDOW,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SILVER-PREP] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Intermediate input path (produced by cleaning.py)
CLEAN_INTERMEDIATE = SILVER_OUTPUT.replace(
    "pamap2_clean.parquet", "pamap2_cleaned_intermediate.parquet"
)

EXCLUDE_FROM_NORM = {"timestamp", "activity_id", "subject_id", "session_type"}


def run():
    """Execute the Silver preprocessing pipeline."""

    log.info("=" * 60)
    log.info("SILVER LAYER — Preprocessing")
    log.info("=" * 60)

    spark = (
        SparkSession.builder
        .appName("PAMAP2_Silver_Preprocessing")
        .master("local[*]")
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.pyspark.python", sys.executable)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    log.info(f"Spark version: {spark.version}")

    # If the intermediate parquet is missing but the final Silver output exists,
    # verify it and exit early without re-running the full pipeline.
    if not os.path.exists(CLEAN_INTERMEDIATE):
        if os.path.exists(SILVER_OUTPUT):
            log.warning(
                f"Intermediate parquet not found: {CLEAN_INTERMEDIATE}"
            )
            log.warning(
                "Final Silver output already exists — verifying existing output."
            )
            df_verify = spark.read.parquet(SILVER_OUTPUT)
            row_count = df_verify.count()
            log.info(f"  Rows    : {row_count:,}")
            log.info(f"  Columns : {len(df_verify.columns)}")
            log.info(f"  Subjects: {df_verify.select('subject_id').distinct().count()}")
            hr_nulls = df_verify.filter(col("heart_rate").isNull()).count()
            log.info(f"  HR nulls: {hr_nulls}")
            assert hr_nulls == 0, f"HR nulls found in existing Silver output: {hr_nulls}"
            log.info("Silver preprocessing already complete — skipping re-run.")
            spark.stop()
            return
        else:
            raise FileNotFoundError(
                f"Neither intermediate nor final Silver parquet found.\n"
                f"Run silver/cleaning.py first to create: {CLEAN_INTERMEDIATE}"
            )

    log.info(f"Reading cleaned intermediate from: {CLEAN_INTERMEDIATE}")
    df_clean = spark.read.parquet(CLEAN_INTERMEDIATE)
    log.info(f"Loaded {df_clean.count():,} rows x {len(df_clean.columns)} columns")

    # HR gap between consecutive readings is ~11 rows at 100 Hz / 9 Hz.
    # HR_FILL_WINDOW=15 covers that gap with margin.
    # Bounded windows (vs unbounded) avoid a full-partition sort scan.
    log.info(f"Interpolating heart rate (fill window = {HR_FILL_WINDOW} rows)")

    win_fwd = (
        Window
        .partitionBy("subject_id", "session_type")
        .orderBy("timestamp")
        .rowsBetween(-HR_FILL_WINDOW, 0)
    )
    win_bwd = (
        Window
        .partitionBy("subject_id", "session_type")
        .orderBy("timestamp")
        .rowsBetween(0, HR_FILL_WINDOW)
    )

    df_clean = df_clean.withColumn(
        "heart_rate",
        last("heart_rate", ignorenulls=True).over(win_fwd)
    )

    df_clean = df_clean.withColumn(
        "heart_rate",
        first("heart_rate", ignorenulls=True).over(win_bwd)
    )

    # Per-subject mean fallback via broadcast join (9 rows × 2 cols — too small
    # to justify a shuffle; broadcast() eliminates the exchange stage entirely).
    hr_means = (
        df_clean
        .groupBy("subject_id")
        .agg(F_mean("heart_rate").alias("_hr_mean"))
    )
    df_clean = (
        df_clean
        .join(broadcast(hr_means), on="subject_id", how="left")
        .withColumn(
            "heart_rate",
            when(col("heart_rate").isNull(), col("_hr_mean"))
            .otherwise(col("heart_rate"))
        )
        .drop("_hr_mean")
    )

    remaining_hr_nulls = df_clean.filter(col("heart_rate").isNull()).count()
    log.info(f"Heart rate nulls remaining: {remaining_hr_nulls}")
    assert remaining_hr_nulls == 0, f"HR interpolation incomplete: {remaining_hr_nulls} nulls"

    sensor_cols = [
        c for c in df_clean.columns
        if c not in EXCLUDE_FROM_NORM
        and df_clean.schema[c].dataType == DoubleType()
    ]
    log.info(f"Normalising {len(sensor_cols)} sensor columns to [0, 1]")

    # spark_min/spark_max propagate NaN (unlike null), so convert NaN to null
    # before computing stats. All 40 columns are normalised in one select() pass.
    agg_exprs = []
    for c in sensor_cols:
        safe_col = when(~isnan(col(c)), col(c))
        agg_exprs.append(spark_min(safe_col).alias(f"{c}__min"))
        agg_exprs.append(spark_max(safe_col).alias(f"{c}__max"))

    stats_row = df_clean.agg(*agg_exprs).first()

    df_normalised = df_clean
    for c in sensor_cols:
        c_min = stats_row[f"{c}__min"]
        c_max = stats_row[f"{c}__max"]
        if c_min is not None and c_max is not None and c_max != c_min:
            df_normalised = df_normalised.withColumn(
                c,
                (col(c) - lit(c_min)) / lit(c_max - c_min),
            )
        else:
            # Constant or all-null column
            df_normalised = df_normalised.withColumn(c, lit(0.0))

    log.info("Post-normalisation stats (sample columns):")
    check_cols = ["heart_rate", "hand_acc_16g_x", "chest_gyro_x", "ankle_mag_z"]
    df_normalised.select(check_cols).summary("min", "max", "count").show()

    log.info(f"Writing Silver parquet to: {SILVER_OUTPUT}")
    os.makedirs(os.path.dirname(SILVER_OUTPUT), exist_ok=True)

    (
        df_normalised
        .repartition("subject_id")
        .write
        .mode("overwrite")
        .partitionBy("subject_id")
        .parquet(SILVER_OUTPUT)
    )

    df_verify = spark.read.parquet(SILVER_OUTPUT)
    log.info(f"Verification:")
    log.info(f"  Rows    : {df_verify.count():,}")
    log.info(f"  Columns : {len(df_verify.columns)}")
    log.info(f"  Subjects: {df_verify.select('subject_id').distinct().count()}")
    log.info("Silver preprocessing complete.")

    spark.stop()


if __name__ == "__main__":
    run()
