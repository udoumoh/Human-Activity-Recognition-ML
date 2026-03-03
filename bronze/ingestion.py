"""
Bronze Layer — PAMAP2 Raw Data Ingestion (Distributed).

This script implements the first stage of the Medallion Architecture:
loading raw .dat sensor files from the PAMAP2 dataset into a typed
Spark DataFrame, preserving the original data as-is, and persisting
it as Parquet for downstream processing.

Medallion Architecture Context
------------------------------
The Bronze layer holds **raw, unprocessed data** exactly as it arrived
from the source. No cleaning, imputation, or feature engineering is
applied here. This preserves data lineage and allows the Silver layer
to be re-run independently if cleaning logic changes.

Distributed Ingestion Strategy
------------------------------
- Spark's CSV reader distributes file parsing across executor cores,
  reading each .dat file as a partition in parallel.
- An explicit StructType schema is passed to the reader so that type
  casting happens inside the distributed CSV parser (no separate
  map/filter RDD pass needed — the DataFrame API + Catalyst optimizer
  handles this more efficiently than RDD-based approaches).
- `input_file_name()` extracts the subject ID from the file path in
  a distributed UDF-free manner (pure SQL expression), avoiding the
  overhead of Python UDFs across partitions.
- The output is written as Parquet partitioned by subject_id, which
  enables partition pruning in downstream queries (e.g., when the
  Silver layer processes one subject at a time).

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

# ── Project imports ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bronze.raw_data_schema import PAMAP2_SCHEMA
from config.settings import (
    PROTOCOL_PATH, OPTIONAL_PATH, BRONZE_OUTPUT,
    SPARK_DRIVER_MEMORY, SPARK_SHUFFLE_PARTITIONS,
)

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BRONZE] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# Helper functions
# ════════════════════════════════════════════════════════════

def list_dat_files(folder: str) -> list:
    """
    Return sorted list of absolute .dat file paths in a folder.

    We use Python's os.listdir() instead of Hadoop glob patterns
    because the native Hadoop glob can fail on Windows with
    UnsatisfiedLinkError on NativeIO.
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

    Distributed processing notes:
    - spark.read.csv() distributes file parsing across cores.
    - Each .dat file becomes at least one Spark partition.
    - input_file_name() runs inside the JVM (no Python UDF overhead).
    - regexp_extract() is a native Catalyst expression executed on
      each partition without serialising data back to the driver.
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

    # Extract subject ID from file path (e.g., "subject101" → 101)
    # This is a pure SQL expression — no Python UDF serialisation
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


# ════════════════════════════════════════════════════════════
# Main ingestion pipeline
# ════════════════════════════════════════════════════════════

def run():
    """Execute the Bronze ingestion pipeline."""

    log.info("=" * 60)
    log.info("BRONZE LAYER — Raw Data Ingestion")
    log.info("=" * 60)

    # ── 1. Initialise Spark session ──────────────────────────
    # local[*] uses all available CPU cores for parallel parsing
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

    # ── 2. Distributed file loading ──────────────────────────
    # Each .dat file is parsed in parallel by Spark executors.
    # The schema enforces types at parse time (no post-hoc casting).
    df_protocol = load_session(spark, PROTOCOL_PATH, "protocol")
    df_optional = load_session(spark, OPTIONAL_PATH, "optional")

    # Union both sessions — unionByName ensures column alignment
    df_raw = df_protocol.unionByName(df_optional)

    proto_count = df_protocol.count()
    opt_count = df_optional.count()
    total_count = proto_count + opt_count

    log.info(f"Protocol rows : {proto_count:,}")
    log.info(f"Optional rows : {opt_count:,}")
    log.info(f"Combined rows : {total_count:,}")
    log.info(f"Columns       : {len(df_raw.columns)}")

    # ── 3. Data validation at ingestion (Bronze-level) ───────
    # Validate structural integrity before any Silver processing.
    # These checks ensure the raw data loaded correctly and meets
    # the expected schema/volume before expensive downstream steps.
    log.info("Running Bronze-level data validation...")

    # Check 1: Schema — expected 56 raw columns + subject_id + session_type = 58
    expected_min_cols = 56
    actual_cols = len(df_raw.columns)
    assert actual_cols >= expected_min_cols, (
        f"Schema mismatch: expected >= {expected_min_cols} columns, got {actual_cols}"
    )
    log.info(f"  Schema check      : {actual_cols} columns (OK)")

    # Check 2: Row count — PAMAP2 full dataset ~3.85 M rows
    EXPECTED_MIN_ROWS = 3_000_000
    EXPECTED_MAX_ROWS = 5_000_000
    assert EXPECTED_MIN_ROWS <= total_count <= EXPECTED_MAX_ROWS, (
        f"Row count {total_count:,} outside expected range "
        f"[{EXPECTED_MIN_ROWS:,}, {EXPECTED_MAX_ROWS:,}]"
    )
    log.info(f"  Row count check   : {total_count:,} in [{EXPECTED_MIN_ROWS:,}, {EXPECTED_MAX_ROWS:,}] (OK)")

    # Check 3: No null subject IDs (file path parsing worked)
    null_subjects = df_raw.filter(col("subject_id").isNull()).count()
    assert null_subjects == 0, f"Found {null_subjects} rows with null subject_id"
    log.info(f"  subject_id nulls  : {null_subjects} (OK)")

    # Check 4: Null percentage for critical sensor columns
    # Heart rate is expected to be ~91% null (9 Hz vs 100 Hz IMU)
    # IMU accelerometer columns should be < 5% null
    hr_null_pct = df_raw.filter(col("heart_rate").isNull()).count() / total_count
    imu_null_pct = df_raw.filter(col("hand_acc_16g_x").isNull()).count() / total_count
    assert hr_null_pct < 0.95, f"Heart rate null % unexpectedly high: {hr_null_pct:.1%}"
    assert imu_null_pct < 0.10, f"IMU accelerometer null % too high: {imu_null_pct:.1%}"
    log.info(f"  heart_rate null % : {hr_null_pct:.1%} (expected ~91% due to 9 Hz sampling)")
    log.info(f"  IMU acc null %    : {imu_null_pct:.1%} (OK)")

    # Check 5: Expected number of subjects
    n_subjects = df_raw.select("subject_id").distinct().count()
    assert n_subjects >= 8, f"Expected >= 8 subjects, found {n_subjects}"
    log.info(f"  Distinct subjects : {n_subjects} (OK)")

    # Check: activity distribution
    log.info("  Activity distribution:")
    df_raw.groupBy("activity_id").count().orderBy("activity_id").show(25, truncate=False)

    # Check: subject × session breakdown
    log.info("  Subject × session breakdown:")
    df_raw.groupBy("subject_id", "session_type").count().orderBy(
        "subject_id", "session_type"
    ).show(20)

    # ── 4. Persist as Parquet (partitioned by subject_id) ────
    # Partitioning strategy:
    #   - Partitioning by subject_id creates 9 partition directories.
    #   - This enables partition pruning: downstream queries that
    #     filter on subject_id only read the relevant partitions.
    #   - repartition("subject_id") ensures one file per subject,
    #     avoiding many small files within each partition directory.
    log.info(f"Writing Bronze Parquet to: {BRONZE_OUTPUT}")

    os.makedirs(os.path.dirname(BRONZE_OUTPUT), exist_ok=True)

    (
        df_raw
        .repartition("subject_id")
        .write
        .mode("overwrite")
        .partitionBy("subject_id")
        .parquet(BRONZE_OUTPUT)
    )

    # ── 5. Verify the write ──────────────────────────────────
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
