"""
Silver Layer — Data Cleaning (Distributed).

Removes invalid rows and columns from Bronze raw data:

1. activity_id == 0: unlabelled transition periods (per PAMAP2 readme).
2. Orientation quaternion columns (12 total): marked invalid in PAMAP2 readme.
3. Rows with total sensor dropout (all three IMU accelerometers null simultaneously).

Usage
-----
    python -m silver.cleaning

Input
-----
    data/bronze/pamap2_raw.parquet

Output
------
    data/silver/pamap2_cleaned_intermediate.parquet
"""

import os
import sys
import logging

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, broadcast
from pyspark.sql.types import IntegerType, StringType, StructType, StructField

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    BRONZE_OUTPUT, SILVER_OUTPUT,
    SPARK_DRIVER_MEMORY, SPARK_SHUFFLE_PARTITIONS,
)

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

    log.info(f"Reading Bronze parquet from: {BRONZE_OUTPUT}")
    df_raw = spark.read.parquet(BRONZE_OUTPUT)
    raw_count = df_raw.count()
    raw_cols = len(df_raw.columns)
    log.info(f"Bronze rows    : {raw_count:,}")
    log.info(f"Bronze columns : {raw_cols}")

    df_clean = df_raw.filter(col("activity_id") != 0)
    after_transient = df_clean.count()
    log.info(
        f"After removing transient (id=0): {after_transient:,} rows "
        f"(dropped {raw_count - after_transient:,})"
    )

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

    # If ALL three key accelerometers are null, the entire sensor suite failed.
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

    log.info("Data quality check after cleaning:")

    null_exprs = [
        count(when(col(c).isNull(), c)).alias(c)
        for c in ["heart_rate", "hand_temperature",
                   "chest_temperature", "ankle_temperature"]
    ]
    log.info("  Null counts for key columns:")
    df_clean.select(null_exprs).show(truncate=False)

    log.info("  Activity distribution (post-cleaning):")
    df_clean.groupBy("activity_id").count().orderBy("activity_id").show(25, truncate=False)

    # 18-row lookup table broadcast to all executors — eliminates shuffle
    # that a SortMergeJoin on the multi-million-row DataFrame would require.
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

    df_clean = df_clean.join(broadcast(label_df), on="activity_id", how="left")
    log.info(
        "Broadcast join applied: activity_name column added "
        f"({df_clean.filter(col('activity_name').isNull()).count()} null names)"
    )

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

    df_verify = spark.read.parquet(CLEAN_INTERMEDIATE)
    log.info(f"Verification: {df_verify.count():,} rows, {len(df_verify.columns)} columns")
    log.info("Silver cleaning complete.")

    spark.stop()


if __name__ == "__main__":
    run()
