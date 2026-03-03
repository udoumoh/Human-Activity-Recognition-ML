"""
Gold Layer — Sliding-Window Feature Engineering (Distributed).

This script implements the first stage of the Gold layer: transforming
the cleaned Silver time-series data into a machine-learning-ready
feature matrix using sliding-window aggregation.

Medallion Architecture Context
------------------------------
The Gold layer produces **business-level / analytics-ready** datasets.
In this pipeline, that means converting 2.7M raw sensor rows into
~5,400 windowed feature rows — each representing a 5-second activity
segment with 172 statistical features.

Feature Extraction Strategy (Distributed)
-----------------------------------------
1. **Window assignment**: For each (subject, activity) group, rows are
   assigned to fixed-duration windows based on their timestamp offset.
   This uses Spark Window functions partitioned by group columns,
   executed across partitions in parallel.

2. **Aggregation**: Per window, we compute 4 statistics (mean, std,
   min, max) for each of 40 sensor columns = 160 features, plus 12
   Signal Magnitude Area (SMA) features for triaxial sensor groups,
   totalling 172 features. All 173 aggregation expressions (172
   features + sample_count) are computed in a SINGLE groupBy().agg()
   pass — Spark's Catalyst optimizer fuses them into one physical
   stage to minimise data shuffling.

3. **Quality gate**: Windows with fewer than 50% of the expected
   samples (< 250 of 500) are discarded. These occur at activity
   boundaries and provide unreliable statistics.

Why groupBy().agg() over RDD mapPartitions
------------------------------------------
The DataFrame groupBy/agg approach allows Catalyst to:
- Push aggregation partially into each partition (partial aggregation)
- Use Tungsten's binary columnar format (no Python serialisation)
- Automatically select hash-based or sort-based aggregation strategy
An RDD-based approach would require manual partial aggregation logic
and Python-level row serialisation between map and reduce steps.

Usage
-----
    python -m gold.feature_engineering

Input
-----
    data/silver/pamap2_clean.parquet

Output
------
    data/gold/pamap2_features.parquet   (partitioned by subject_id)
"""

import os
import sys
import logging

from pyspark.sql import SparkSession, Window
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import (
    col, lit, floor, abs as spark_abs,
    count,
    mean   as F_mean,
    stddev as F_stddev,
    min    as F_min,
    max    as F_max,
)

# ── Project imports ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    SILVER_OUTPUT, GOLD_FEATURES_OUTPUT,
    SPARK_DRIVER_MEMORY, SPARK_SHUFFLE_PARTITIONS,
    WINDOW_DURATION_SEC, SAMPLE_RATE_HZ, MIN_WINDOW_FILL,
    META_COLS, IMU_LOCATIONS,
)

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GOLD-FEAT] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Derived constants ────────────────────────────────────────
EXPECTED_SAMPLES = int(WINDOW_DURATION_SEC * SAMPLE_RATE_HZ)   # 500
MIN_SAMPLES = int(EXPECTED_SAMPLES * MIN_WINDOW_FILL)          # 250

TRIAXIAL_SENSORS = ["acc_16g", "acc_6g", "gyro", "mag"]


def run():
    """Execute the Gold feature engineering pipeline."""

    log.info("=" * 60)
    log.info("GOLD LAYER — Sliding-Window Feature Engineering")
    log.info("=" * 60)

    # ── 1. Initialise Spark ──────────────────────────────────
    spark = (
        SparkSession.builder
        .appName("PAMAP2_Gold_FeatureEngineering")
        .master("local[*]")
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    log.info(f"Spark version: {spark.version}")

    # ── 2. Load Silver data ──────────────────────────────────
    log.info(f"Reading Silver parquet from: {SILVER_OUTPUT}")
    df = spark.read.parquet(SILVER_OUTPUT)
    log.info(f"Loaded {df.count():,} rows x {len(df.columns)} columns")

    # ── 3. Identify sensor columns and triaxial groups ───────
    sensor_cols = sorted([
        c for c in df.columns
        if c not in META_COLS
        and isinstance(df.schema[c].dataType, DoubleType)
    ])

    triaxial_groups = [
        (f"{loc}_{sensor}",
         f"{loc}_{sensor}_x", f"{loc}_{sensor}_y", f"{loc}_{sensor}_z")
        for loc in IMU_LOCATIONS
        for sensor in TRIAXIAL_SENSORS
    ]

    log.info(f"Window        : {WINDOW_DURATION_SEC}s "
             f"({EXPECTED_SAMPLES} samples @ {SAMPLE_RATE_HZ} Hz)")
    log.info(f"Min samples   : {MIN_SAMPLES} (reject < {MIN_WINDOW_FILL*100:.0f}% full)")
    log.info(f"Sensor cols   : {len(sensor_cols)}")
    log.info(f"Triaxial SMA  : {len(triaxial_groups)} groups")

    # ── 4. Assign window IDs ────────────────────────────────
    # Windows are computed within each (subject, activity) segment
    # so no window ever straddles two different activities.
    #   window_id = floor((t - t_min) / window_duration)
    seg_window = Window.partitionBy("subject_id", "activity_id")

    df_win = (
        df
        .withColumn("_t0", F_min("timestamp").over(seg_window))
        .withColumn(
            "window_id",
            floor((col("timestamp") - col("_t0")) / lit(WINDOW_DURATION_SEC))
            .cast("long"),
        )
        .drop("_t0")
    )

    # ── 5. Build aggregation expressions ─────────────────────
    # ALL aggregations are collected into a single list so the
    # entire feature extraction runs in ONE groupBy().agg() pass.
    agg_exprs = [count("*").alias("sample_count")]

    # 5a. Per-sensor statistics (mean, std, min, max)
    for c in sensor_cols:
        agg_exprs.extend([
            F_mean(col(c)).alias(f"{c}_mean"),
            F_stddev(col(c)).alias(f"{c}_std"),
            F_min(col(c)).alias(f"{c}_min"),
            F_max(col(c)).alias(f"{c}_max"),
        ])

    # 5b. Signal Magnitude Area (SMA) per triaxial group
    # SMA = mean( |x| + |y| + |z| ) over the window
    for name, x_col, y_col, z_col in triaxial_groups:
        agg_exprs.append(
            F_mean(
                spark_abs(col(x_col))
                + spark_abs(col(y_col))
                + spark_abs(col(z_col))
            ).alias(f"{name}_sma")
        )

    stat_features = len(sensor_cols) * 4
    sma_features = len(triaxial_groups)
    log.info(f"Stat features : {stat_features} ({len(sensor_cols)} cols x 4)")
    log.info(f"SMA features  : {sma_features}")
    log.info(f"Total features: {stat_features + sma_features}")

    # ── 6. Execute windowed aggregation ──────────────────────
    group_keys = ["subject_id", "activity_id", "window_id"]
    df_features_raw = df_win.groupBy(*group_keys).agg(*agg_exprs)

    raw_count = df_features_raw.count()
    log.info(f"Raw feature rows: {raw_count:,}")

    # ── 7. Quality gate — drop incomplete windows ────────────
    df_features = (
        df_features_raw
        .filter(col("sample_count") >= MIN_SAMPLES)
        .drop("window_id", "sample_count")
    )

    final_count = df_features.count()
    dropped = raw_count - final_count
    log.info(f"Windows kept   : {final_count:,}")
    log.info(f"Windows dropped: {dropped:,} (< {MIN_SAMPLES} samples)")
    log.info(f"Final columns  : {len(df_features.columns)} "
             f"({len(df_features.columns) - 2} features + subject_id + activity_id)")

    # Activity distribution
    log.info("Activity distribution (windowed):")
    df_features.groupBy("activity_id").count().orderBy("activity_id").show(
        25, truncate=False
    )

    # ── 8. Save as Parquet ───────────────────────────────────
    log.info(f"Writing Gold features to: {GOLD_FEATURES_OUTPUT}")
    os.makedirs(os.path.dirname(GOLD_FEATURES_OUTPUT), exist_ok=True)

    (
        df_features
        .repartition("subject_id")
        .write
        .mode("overwrite")
        .partitionBy("subject_id")
        .parquet(GOLD_FEATURES_OUTPUT)
    )

    # ── 9. Verify ────────────────────────────────────────────
    df_verify = spark.read.parquet(GOLD_FEATURES_OUTPUT)
    log.info(f"Verification:")
    log.info(f"  Rows    : {df_verify.count():,}")
    log.info(f"  Columns : {len(df_verify.columns)}")
    log.info(f"  Subjects: {df_verify.select('subject_id').distinct().count()}")
    log.info("Gold feature engineering complete.")

    spark.stop()


if __name__ == "__main__":
    run()
