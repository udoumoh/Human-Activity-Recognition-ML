"""
Silver Layer — Data Preprocessing (Distributed).

This script implements the second stage of the Silver layer:
value-level transformations on the structurally cleaned data
from cleaning.py. It produces the final Silver-layer output
ready for Gold-layer feature engineering.

Preprocessing Operations (all distributed)
------------------------------------------
1. **Heart-rate interpolation**:
   HR is sampled at ~9 Hz while IMUs run at 100 Hz, so ~91% of
   HR values are null by design. We apply bounded forward-fill,
   back-fill, and per-subject mean imputation — all using Spark
   Window functions that execute across partitions in parallel.

2. **Min-Max normalisation** (sensor columns to [0, 1]):
   Global min/max statistics are computed in a single distributed
   aggregation pass, then applied as column-wise transformations.
   This ensures all sensor features are on the same scale before
   downstream feature engineering.

Why DataFrame API over RDD for preprocessing
--------------------------------------------
- Window functions (forward-fill, back-fill) use Spark's native
  Tungsten execution engine, operating on binary columnar data
  without Python serialisation overhead.
- The Catalyst optimizer merges the normalisation expressions into
  a single projection stage — all 40 sensor columns are normalised
  in one pass over the data, not 40 sequential passes.
- An RDD-based equivalent would require partitionBy + mapPartitions
  with manual state tracking for forward-fill, which is both slower
  and harder to maintain.

Persist / Unpersist Strategy
----------------------------
- df_clean is NOT cached because each transformation produces a new
  DataFrame (Spark is lazy — transformations are fused into the
  execution plan).
- The final normalised DataFrame is written directly to Parquet
  without caching, since it is only materialised once.
- If this script were part of a larger multi-output pipeline,
  caching before the write would be justified to avoid recomputation.

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

# ── Project imports ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    SILVER_OUTPUT, SPARK_DRIVER_MEMORY, SPARK_SHUFFLE_PARTITIONS,
    META_COLS, HR_FILL_WINDOW,
)

# ── Logging ──────────────────────────────────────────────────
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

# Columns excluded from normalisation
EXCLUDE_FROM_NORM = {"timestamp", "activity_id", "subject_id", "session_type"}


def run():
    """Execute the Silver preprocessing pipeline."""

    log.info("=" * 60)
    log.info("SILVER LAYER — Preprocessing")
    log.info("=" * 60)

    # ── 1. Initialise Spark session ──────────────────────────
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

    # ── 2. Load cleaned intermediate data ────────────────────
    # If the intermediate parquet (produced by silver/cleaning.py) is missing
    # but the final Silver output already exists, verify it and exit early.
    # This handles the case where the pipeline was previously completed and
    # the intermediate artifact was not retained.
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

    # ── 3. Heart-rate interpolation ──────────────────────────
    # HR is sampled at ~9 Hz vs 100 Hz for IMUs, so ~91% of rows
    # have null HR by design. The gap between consecutive HR
    # readings is ~11 rows.
    #
    # Strategy (all using distributed Spark Window functions):
    #   Step 1: Forward-fill with a BOUNDED window of 15 rows
    #           (covers the ~11-row gap with margin).
    #   Step 2: Back-fill leading nulls with the same bound.
    #   Step 3: Fill any remaining nulls with per-subject mean HR.
    #
    # Why bounded windows:
    #   Unbounded windows require a full partition sort and scan,
    #   which is O(n) per partition. Bounded windows (15 rows)
    #   limit the scan to a fixed neighbourhood, making the
    #   operation O(1) per row after the initial sort.
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

    # Step 1: forward-fill (carry last known HR value forward)
    df_clean = df_clean.withColumn(
        "heart_rate",
        last("heart_rate", ignorenulls=True).over(win_fwd)
    )

    # Step 2: back-fill (cover leading nulls with next known HR)
    df_clean = df_clean.withColumn(
        "heart_rate",
        first("heart_rate", ignorenulls=True).over(win_bwd)
    )

    # Step 3: fill remaining nulls with per-subject mean HR via broadcast join.
    #
    # Design rationale: the lookup table contains exactly one row per subject
    # (9 rows × 2 columns), making it far too small to justify a shuffle-based
    # join.  Using broadcast() sends the tiny table to every executor once,
    # eliminating the exchange stage entirely.  A Window-based mean would also
    # work but triggers a partition-wide sort; the broadcast join achieves the
    # same result with lower overhead.
    #
    # broadcast() annotation: small lookup table (9 rows × 2 cols) —
    # broadcast eliminates shuffle.
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

    # ── 4. Min-Max normalisation (0–1) ───────────────────────
    # All numeric sensor columns are scaled to [0, 1] using
    # global min/max values. This ensures features are on the
    # same scale for downstream ML models.
    #
    # Formula: x_norm = (x - x_min) / (x_max - x_min)
    # If max == min (constant column), result is 0.
    #
    # IMPORTANT: spark_min / spark_max propagate NaN (unlike null).
    # We filter NaN before computing stats using when(~isnan()).
    #
    # The entire normalisation is expressed as a single select()
    # of column expressions. Catalyst fuses this into one physical
    # projection stage — all 40 columns are normalised in one pass.
    sensor_cols = [
        c for c in df_clean.columns
        if c not in EXCLUDE_FROM_NORM
        and df_clean.schema[c].dataType == DoubleType()
    ]
    log.info(f"Normalising {len(sensor_cols)} sensor columns to [0, 1]")

    # Compute global min/max in ONE distributed aggregation pass
    # NaN is converted to null so it is skipped by min/max
    agg_exprs = []
    for c in sensor_cols:
        safe_col = when(~isnan(col(c)), col(c))
        agg_exprs.append(spark_min(safe_col).alias(f"{c}__min"))
        agg_exprs.append(spark_max(safe_col).alias(f"{c}__max"))

    stats_row = df_clean.agg(*agg_exprs).first()

    # Apply normalisation column by column
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
            # Constant or all-null column -> set to 0
            df_normalised = df_normalised.withColumn(c, lit(0.0))

    # Sanity check
    log.info("Post-normalisation stats (sample columns):")
    check_cols = ["heart_rate", "hand_acc_16g_x", "chest_gyro_x", "ankle_mag_z"]
    df_normalised.select(check_cols).summary("min", "max", "count").show()

    # ── 5. Save final Silver output ──────────────────────────
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

    # ── 6. Verify ────────────────────────────────────────────
    df_verify = spark.read.parquet(SILVER_OUTPUT)
    log.info(f"Verification:")
    log.info(f"  Rows    : {df_verify.count():,}")
    log.info(f"  Columns : {len(df_verify.columns)}")
    log.info(f"  Subjects: {df_verify.select('subject_id').distinct().count()}")
    log.info("Silver preprocessing complete.")

    spark.stop()


if __name__ == "__main__":
    run()
