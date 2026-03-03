"""
Silver Layer — Data Cleaning (Distributed).

This script implements the first stage of the Silver layer: removing
invalid rows and columns from the Bronze raw data to produce a
structurally sound dataset ready for preprocessing.

Medallion Architecture Context
------------------------------
The Silver layer transforms raw Bronze data into a **validated,
cleaned** dataset. This script handles structural issues — dropping
rows and columns that are invalid per the PAMAP2 documentation —
while the companion preprocessing.py handles value-level transforms
(interpolation, normalisation).

Cleaning Operations (all distributed via DataFrame API)
-------------------------------------------------------
1. **Remove transient activities** (activity_id == 0):
   The PAMAP2 readme specifies that ID 0 represents unlabelled
   transition periods between activities. These are not valid
   training targets and must be discarded.

2. **Drop invalid orientation columns** (12 columns):
   The PAMAP2 readme states that the 4 orientation quaternion
   values per IMU are invalid / unreliable. Removing them here
   reduces the column count from 56 to 44 and prevents downstream
   feature engineering from computing statistics on garbage data.

3. **Drop complete sensor dropout rows**:
   Rows where all three IMU placements have null accelerometer
   readings indicate a total sensor failure at that timestamp.
   These rows carry no usable information.

Why DataFrame API over RDD
--------------------------
All three cleaning operations are expressed as filter() and drop()
calls on DataFrames. Spark's Catalyst optimizer fuses these into a
single physical plan stage — the data is read once and all three
filters are applied in a single pass. An equivalent RDD pipeline
would require explicit map/filter chains with manual serialisation
between steps.

Usage
-----
    python -m silver.cleaning

Input
-----
    data/bronze/pamap2_raw.parquet

Output
------
    data/silver/pamap2_cleaned_intermediate.parquet
    (intermediate — consumed by silver.preprocessing)
"""

import os
import sys
import logging

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, broadcast
from pyspark.sql.types import IntegerType, StringType, StructType, StructField

# ── Project imports ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    BRONZE_OUTPUT, SILVER_OUTPUT,
    SPARK_DRIVER_MEMORY, SPARK_SHUFFLE_PARTITIONS,
)

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SILVER-CLEAN] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Intermediate output path (between cleaning and preprocessing)
CLEAN_INTERMEDIATE = SILVER_OUTPUT.replace(
    "pamap2_clean.parquet", "pamap2_cleaned_intermediate.parquet"
)


def run():
    """Execute the Silver cleaning pipeline."""

    log.info("=" * 60)
    log.info("SILVER LAYER — Data Cleaning")
    log.info("=" * 60)

    # ── 1. Initialise Spark session ──────────────────────────
    spark = (
        SparkSession.builder
        .appName("PAMAP2_Silver_Cleaning")
        .master("local[*]")
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    log.info(f"Spark version: {spark.version}")

    # ── 2. Load Bronze data ──────────────────────────────────
    log.info(f"Reading Bronze parquet from: {BRONZE_OUTPUT}")
    df_raw = spark.read.parquet(BRONZE_OUTPUT)
    raw_count = df_raw.count()
    raw_cols = len(df_raw.columns)
    log.info(f"Bronze rows    : {raw_count:,}")
    log.info(f"Bronze columns : {raw_cols}")

    # ── 3a. Remove transient activities (activity_id == 0) ───
    # The PAMAP2 readme specifies that activity_id 0 represents
    # unlabelled transitions. These rows are filtered out in a
    # distributed fashion — Spark applies the predicate across
    # all partitions in parallel without collecting to the driver.
    df_clean = df_raw.filter(col("activity_id") != 0)
    after_transient = df_clean.count()
    log.info(
        f"After removing transient (id=0): {after_transient:,} rows "
        f"(dropped {raw_count - after_transient:,})"
    )

    # ── 3b. Drop 12 invalid orientation columns ──────────────
    # The PAMAP2 readme marks orientation quaternion values as
    # invalid for all three IMU placements. Dropping them here
    # prevents downstream feature engineering from computing
    # statistics on unreliable data.
    orient_cols = [
        f"{loc}_orient_{i}"
        for loc in ("hand", "chest", "ankle")
        for i in range(4)
    ]
    df_clean = df_clean.drop(*orient_cols)
    log.info(
        f"Dropped {len(orient_cols)} invalid orientation columns "
        f"-> {len(df_clean.columns)} columns remain"
    )

    # ── 3c. Drop rows with complete sensor dropout ───────────
    # If ALL three key accelerometers are null at a timestamp,
    # the entire sensor suite failed — the row is unrecoverable.
    # This filter runs distributed across all partitions.
    df_clean = df_clean.filter(
        col("hand_acc_16g_x").isNotNull()
        | col("chest_acc_16g_x").isNotNull()
        | col("ankle_acc_16g_x").isNotNull()
    )
    after_dropout = df_clean.count()
    log.info(
        f"After dropping sensor-dropout rows: {after_dropout:,} rows "
        f"(dropped {after_transient - after_dropout:,})"
    )

    # ── 4. Data quality report ───────────────────────────────
    log.info("Data quality check after cleaning:")

    # Null counts for key columns
    null_exprs = [
        count(when(col(c).isNull(), c)).alias(c)
        for c in ["heart_rate", "hand_temperature",
                   "chest_temperature", "ankle_temperature"]
    ]
    log.info("  Null counts for key columns:")
    df_clean.select(null_exprs).show(truncate=False)

    # Activity distribution after cleaning
    log.info("  Activity distribution (post-cleaning):")
    df_clean.groupBy("activity_id").count().orderBy("activity_id").show(25, truncate=False)

    # ── 4b. Broadcast join: annotate with activity labels ────
    # PAMAP2 activity ID → human-readable name lookup.
    # This small 18-row table is broadcast to all executors so
    # that no shuffle is needed when joining against the large
    # (multi-million row) cleaned DataFrame.
    # The broadcast() hint instructs Catalyst to use a BroadcastHashJoin
    # rather than a SortMergeJoin, eliminating the shuffle exchange stage.
    activity_meta = [
        (1, "Lying"), (2, "Sitting"), (3, "Standing"), (4, "Walking"),
        (5, "Running"), (6, "Cycling"), (7, "Nordic Walking"),
        (9, "Watching TV"), (10, "Computer Work"), (11, "Car Driving"),
        (12, "Ascending Stairs"), (13, "Descending Stairs"),
        (16, "Vacuum Cleaning"), (17, "Ironing"), (18, "Folding Laundry"),
        (19, "House Cleaning"), (20, "Playing Soccer"), (24, "Rope Jumping"),
    ]
    schema = StructType([
        StructField("activity_id", IntegerType(), False),
        StructField("activity_name", StringType(), False),
    ])
    label_df = spark.createDataFrame(activity_meta, schema=schema)

    # broadcast() pins label_df in executor memory — safe because it is
    # tiny (18 rows). The join adds activity_name for lineage traceability.
    df_clean = df_clean.join(broadcast(label_df), on="activity_id", how="left")
    log.info(
        "Broadcast join applied: activity_name column added "
        f"({df_clean.filter(col('activity_name').isNull()).count()} null names)"
    )

    # ── 5. Persist intermediate output ───────────────────────
    # We save the cleaned (but not yet normalised) data as an
    # intermediate parquet. This allows preprocessing.py to be
    # re-run independently without repeating the cleaning step.
    #
    # Persist/unpersist strategy:
    #   - We do NOT call .cache() here because the DataFrame is
    #     written once and not reused in this script.
    #   - Caching would waste driver memory for no benefit.
    #   - The downstream preprocessing.py will read from disk
    #     and can cache as needed for its multi-pass operations.
    log.info(f"Writing cleaned intermediate to: {CLEAN_INTERMEDIATE}")
    os.makedirs(os.path.dirname(CLEAN_INTERMEDIATE), exist_ok=True)

    (
        df_clean
        .repartition("subject_id")
        .write
        .mode("overwrite")
        .partitionBy("subject_id")
        .parquet(CLEAN_INTERMEDIATE)
    )

    # ── 6. Verify ────────────────────────────────────────────
    df_verify = spark.read.parquet(CLEAN_INTERMEDIATE)
    log.info(f"Verification: {df_verify.count():,} rows, {len(df_verify.columns)} columns")
    log.info("Silver cleaning complete.")

    spark.stop()


if __name__ == "__main__":
    run()
