"""
Bronze Layer — PAMAP2 Raw Data Ingestion (Distributed).

Loads raw .dat sensor files into a typed Spark DataFrame and persists
as Parquet partitioned by subject_id for downstream Silver processing.

Usage
-----
    python -m bronze.ingestion

Output
------
    data/bronze/pamap2_raw.parquet   (partitioned by subject_id)
"""

import os
import sys
import logging

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import (
    col, lit, input_file_name, regexp_extract, count, when,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bronze.raw_data_schema import PAMAP2_SCHEMA
from config.settings import (
    PROTOCOL_PATH, OPTIONAL_PATH, BRONZE_OUTPUT,
    SPARK_DRIVER_MEMORY, SPARK_SHUFFLE_PARTITIONS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BRONZE] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def list_dat_files(folder: str) -> list:
    """
    Return sorted list of absolute .dat file paths in a folder.

    Uses os.listdir() instead of Hadoop glob patterns because native
    Hadoop glob can fail on Windows with UnsatisfiedLinkError on NativeIO.
    """
    return sorted([
        os.path.join(folder, f).replace("\\", "/")
        for f in os.listdir(folder)
        if f.endswith(".dat")
    ])


def load_session(spark, folder_path: str, session_type: str):
    """
    Read every .dat file in *folder_path* using the PAMAP2 schema,
    then add subject_id (extracted from filename) and session_type.
    """
    file_list = list_dat_files(folder_path)
    log.info(f"{session_type}: found {len(file_list)} .dat files")

    df = (
        spark.read
        .schema(PAMAP2_SCHEMA)
        .option("header", "false")
        .option("delimiter", " ")
        .option("nanValue", "NaN")
        .csv(file_list)
    )

    # Extract subject ID from file path (e.g., "subject101" → 101) via
    # pure SQL expression — no Python UDF serialisation overhead.
    df = (
        df
        .withColumn("_path", input_file_name())
        .withColumn(
            "subject_id",
            regexp_extract("_path", r"subject(\d+)", 1).cast(IntegerType()),
        )
        .withColumn("session_type", lit(session_type))
        .drop("_path")
    )
    return df


def run():
    """Execute the Bronze ingestion pipeline."""

    log.info("=" * 60)
    log.info("BRONZE LAYER — Raw Data Ingestion")
    log.info("=" * 60)

    spark = (
        SparkSession.builder
        .appName("PAMAP2_Bronze_Ingestion")
        .master("local[*]")
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    log.info(f"Spark version : {spark.version}")
    log.info(f"Parallelism   : {spark.sparkContext.defaultParallelism}")

    df_protocol = load_session(spark, PROTOCOL_PATH, "protocol")
    df_optional = load_session(spark, OPTIONAL_PATH, "optional")

    df_raw = df_protocol.unionByName(df_optional)

    proto_count = df_protocol.count()
    opt_count = df_optional.count()
    total_count = proto_count + opt_count

    log.info(f"Protocol rows : {proto_count:,}")
    log.info(f"Optional rows : {opt_count:,}")
    log.info(f"Combined rows : {total_count:,}")
    log.info(f"Columns       : {len(df_raw.columns)}")

    log.info("Running Bronze-level data validation...")

    # Expected 56 raw columns + subject_id + session_type = 58
    expected_min_cols = 56
    actual_cols = len(df_raw.columns)
    assert actual_cols >= expected_min_cols, (
        f"Schema mismatch: expected >= {expected_min_cols} columns, got {actual_cols}"
    )
    log.info(f"  Schema check      : {actual_cols} columns (OK)")

    # PAMAP2 full dataset is ~3.85 M rows
    EXPECTED_MIN_ROWS = 3_000_000
    EXPECTED_MAX_ROWS = 5_000_000
    assert EXPECTED_MIN_ROWS <= total_count <= EXPECTED_MAX_ROWS, (
        f"Row count {total_count:,} outside expected range "
        f"[{EXPECTED_MIN_ROWS:,}, {EXPECTED_MAX_ROWS:,}]"
    )
    log.info(f"  Row count check   : {total_count:,} in [{EXPECTED_MIN_ROWS:,}, {EXPECTED_MAX_ROWS:,}] (OK)")

    null_subjects = df_raw.filter(col("subject_id").isNull()).count()
    assert null_subjects == 0, f"Found {null_subjects} rows with null subject_id"
    log.info(f"  subject_id nulls  : {null_subjects} (OK)")

    # Heart rate is ~91% null by design (9 Hz vs 100 Hz IMU sampling rate)
    hr_null_pct = df_raw.filter(col("heart_rate").isNull()).count() / total_count
    imu_null_pct = df_raw.filter(col("hand_acc_16g_x").isNull()).count() / total_count
    assert hr_null_pct < 0.95, f"Heart rate null % unexpectedly high: {hr_null_pct:.1%}"
    assert imu_null_pct < 0.10, f"IMU accelerometer null % too high: {imu_null_pct:.1%}"
    log.info(f"  heart_rate null % : {hr_null_pct:.1%} (expected ~91% due to 9 Hz sampling)")
    log.info(f"  IMU acc null %    : {imu_null_pct:.1%} (OK)")

    n_subjects = df_raw.select("subject_id").distinct().count()
    assert n_subjects >= 8, f"Expected >= 8 subjects, found {n_subjects}"
    log.info(f"  Distinct subjects : {n_subjects} (OK)")

    log.info("  Activity distribution:")
    df_raw.groupBy("activity_id").count().orderBy("activity_id").show(25, truncate=False)

    log.info("  Subject × session breakdown:")
    df_raw.groupBy("subject_id", "session_type").count().orderBy(
        "subject_id", "session_type"
    ).show(20)

    log.info(f"Writing Bronze Parquet to: {BRONZE_OUTPUT}")

    os.makedirs(os.path.dirname(BRONZE_OUTPUT), exist_ok=True)

    # repartition("subject_id") ensures one file per subject directory,
    # enabling partition pruning in downstream Silver queries.
    (
        df_raw
        .repartition("subject_id")
        .write
        .mode("overwrite")
        .partitionBy("subject_id")
        .parquet(BRONZE_OUTPUT)
    )

    df_verify = spark.read.parquet(BRONZE_OUTPUT)
    verify_count = df_verify.count()
    verify_cols = len(df_verify.columns)
    verify_subjects = df_verify.select("subject_id").distinct().count()

    log.info(f"Verification:")
    log.info(f"  Parquet rows   : {verify_count:,}")
    log.info(f"  Parquet columns: {verify_cols}")
    log.info(f"  Partitions     : {verify_subjects} subjects")

    assert verify_count == total_count, (
        f"Row count mismatch: wrote {total_count:,}, read back {verify_count:,}"
    )
    log.info("Bronze ingestion complete.")

    spark.stop()


if __name__ == "__main__":
    run()
