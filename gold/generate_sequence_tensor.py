"""
Gold Layer — Raw Sequence Tensor Generation (Hybrid Pipeline, Stage 1/2).

Transforms the Silver layer (2.7 M rows of 100 Hz sensor data) into a
3-D NumPy tensor suitable for the 1-D CNN:

    tensor shape : (N_windows, 500, N_channels)
    label shape  : (N_windows,)              [activity_id per window]
    meta CSV     : window_id, subject_id, activity_id

This path retains the full time series (vs. the 172-scalar statistical path
in gold/feature_engineering.py) so the CNN can learn temporal patterns from
raw inertial signals.

Window Strategy
---------------
- Duration  : 5 seconds  (WINDOW_DURATION_SEC = 5.0)
- Timesteps : 500 samples (SAMPLE_RATE_HZ × WINDOW_DURATION_SEC)
- Grouping  : Per (subject_id, activity_id) — windows never straddle two
              activities; avoids label contamination at boundaries.
- Filter    : Only complete windows (exactly 500 timesteps) are retained.
- Window ID : floor((timestamp − t_min) / 5.0) — same formula as
              feature_engineering.py, ensuring consistent segmentation.

Usage
-----
    python -m gold.generate_sequence_tensor
"""

import os
import sys
import time
import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import (
    SILVER_OUTPUT,
    SEQUENCE_WINDOWS_PARQUET, SEQUENCE_TENSOR_NPY,
    SEQUENCE_LABELS_NPY, SEQUENCE_META_CSV,
    SPARK_DRIVER_MEMORY, SPARK_SHUFFLE_PARTITIONS,
    WINDOW_DURATION_SEC, SAMPLE_RATE_HZ, META_COLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SEQ-TENSOR] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

WINDOW_STEPS = int(SAMPLE_RATE_HZ * WINDOW_DURATION_SEC)   # 500 timesteps
REQUIRED_STEPS = WINDOW_STEPS


def get_spark() -> SparkSession:
    """
    SparkSession tuned for the sequence tensor generation job.

    local[4] — the bottleneck is I/O-bound Silver read and the shuffle for
    grouping, so extra threads help throughput without contending for driver memory.
    AQE coalesces shuffle partitions after the groupBy count.
    """
    return (
        SparkSession.builder
        .appName("HAR-SequenceTensor")
        .master("local[4]")
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # 10 MB threshold covers the completeness-filter table (~120 KB max)
        .config("spark.sql.autoBroadcastJoinThreshold", str(10 * 1024 * 1024))
        .getOrCreate()
    )


def identify_sensor_cols(df) -> list:
    """
    Return sorted list of DoubleType columns that are not metadata.

    Excludes META_COLS = {timestamp, activity_id, subject_id, session_type}
    and any StringType columns (e.g., activity_name added by cleaning.py).

    On the PAMAP2 Silver schema this resolves to 40 channels:
        heart_rate (1) + hand_* (13) + chest_* (13) + ankle_* (13) = 40
    """
    exclude = set(META_COLS) | {"activity_name"}
    return sorted([
        f.name for f in df.schema.fields
        if isinstance(f.dataType, DoubleType) and f.name not in exclude
    ])


def assign_windows_and_filter(df, sensor_cols: list):
    """
    Assign 5-second window IDs and retain only complete (500-step) windows.

    Returns a Spark DataFrame with columns:
        subject_id, activity_id, local_window_id, timestamp, *sensor_cols

    Algorithm
    ---------
    1. Partition by (subject_id, activity_id) — one shuffle to co-locate rows.
    2. Compute t_min per group via Window function (no additional shuffle).
    3. local_window_id = floor((t − t_min) / WINDOW_DURATION_SEC)
    4. Count rows per (subject, activity, local_window_id).
       Filter where count == 500 (complete windows only).
    5. Broadcast-join the completeness table back (~130 KB → auto-broadcast).
    """
    group_keys = ["subject_id", "activity_id"]

    grp_window = Window.partitionBy(*group_keys)
    df = (
        df
        .withColumn("_t_min", F.min("timestamp").over(grp_window))
        .withColumn(
            "local_window_id",
            F.floor(
                (F.col("timestamp") - F.col("_t_min")) / F.lit(float(WINDOW_DURATION_SEC))
            ).cast("long"),
        )
        .drop("_t_min")
    )

    # Completeness filter table — tiny, will be auto-broadcast
    win_counts = (
        df
        .groupBy(*group_keys, "local_window_id")
        .agg(F.count("*").alias("_n_steps"))
        .filter(F.col("_n_steps") == REQUIRED_STEPS)
        .drop("_n_steps")
        .hint("broadcast")
    )

    df_complete = df.join(
        win_counts,
        on=[*group_keys, "local_window_id"],
        how="inner",
    )

    select_cols = [*group_keys, "local_window_id", "timestamp"] + sensor_cols
    return df_complete.select(*select_cols)


def write_window_parquet(df, path: str) -> None:
    """
    Write windowed rows to Parquet partitioned by subject_id (9 files).

    Partitioning by subject_id (not window_id) keeps partition count low
    and avoids the many-small-files problem. Rows are sorted within partitions
    by (local_window_id, timestamp) so the driver-side reshape can iterate
    sequentially without resorting.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    log.info(f"Writing windowed Parquet → {path}")
    t0 = time.time()
    (
        df
        .sortWithinPartitions("local_window_id", "timestamp")
        .write
        .mode("overwrite")
        .partitionBy("subject_id")
        .parquet(path)
    )
    log.info(f"Parquet write done in {time.time() - t0:.1f}s")


def parquet_to_numpy(parquet_path: str, sensor_cols: list):
    """
    Read the windowed Parquet and reshape into a 3-D tensor.

    Returns
    -------
    X        : np.ndarray  shape (N_windows, WINDOW_STEPS, n_channels)  float32
    y        : np.ndarray  shape (N_windows,)                            int32
    meta_df  : pd.DataFrame  columns=[window_id, subject_id, activity_id]
    """
    log.info("Reading windowed Parquet via PyArrow …")
    t0 = time.time()

    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    del table  # release Arrow memory before building NumPy tensor

    # Global sort reconciles across the subject_id partition files; within-
    # partition sort from Spark covers most of this already.
    df = df.sort_values(
        ["subject_id", "activity_id", "local_window_id", "timestamp"]
    ).reset_index(drop=True)

    log.info(f"Pandas read done in {time.time() - t0:.1f}s  "
             f"({len(df):,} rows, {df.shape[1]} cols)")

    group_key = ["subject_id", "activity_id", "local_window_id"]
    df["_win_idx"] = df.groupby(group_key, sort=False).ngroup()

    n_windows  = int(df["_win_idx"].max()) + 1
    n_channels = len(sensor_cols)

    log.info(f"Building tensor: {n_windows} windows × "
             f"{WINDOW_STEPS} steps × {n_channels} channels …")

    sensor_vals = df[sensor_cols].values.astype(np.float32)
    win_idx_arr = df["_win_idx"].values
    activity_arr = df["activity_id"].values
    subject_arr  = df["subject_id"].values

    X = np.zeros((n_windows, WINDOW_STEPS, n_channels), dtype=np.float32)
    y = np.zeros(n_windows, dtype=np.int32)
    meta_rows = []

    # searchsorted on sorted win_idx avoids iterating over all rows per window
    boundaries = np.searchsorted(win_idx_arr, np.arange(n_windows + 1))
    for i in range(n_windows):
        s, e = boundaries[i], boundaries[i + 1]
        rows = slice(s, min(s + WINDOW_STEPS, e))
        length = min(WINDOW_STEPS, e - s)
        X[i, :length] = sensor_vals[s:s + length]
        y[i] = activity_arr[s]
        meta_rows.append({
            "window_id":   i,
            "subject_id":  int(subject_arr[s]),
            "activity_id": int(activity_arr[s]),
        })

    log.info(f"Tensor built in {time.time() - t0:.1f}s — shape: {X.shape}")
    return X, y, pd.DataFrame(meta_rows)


def run():
    """End-to-end: Silver → sequence_tensor.npy + sequence_labels.npy."""
    log.info("=" * 65)
    log.info("SEQUENCE TENSOR GENERATION  (Silver → Gold/DL)")
    log.info("=" * 65)

    os.makedirs(os.path.join(PROJECT_ROOT, "data", "gold"), exist_ok=True)
    os.makedirs(PROJECT_ROOT, exist_ok=True)

    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    log.info(f"Loading Silver from {SILVER_OUTPUT} …")
    df = spark.read.parquet(SILVER_OUTPUT)
    log.info(f"Silver rows: {df.count():,}  |  columns: {len(df.columns)}")

    df = df.filter(F.col("activity_id") != 0)

    sensor_cols = identify_sensor_cols(df)
    log.info(f"Sensor channels: {len(sensor_cols)}  "
             f"(e.g. {sensor_cols[:3]} … {sensor_cols[-1]})")

    # Heart rate has ~91% nulls at 100 Hz; zero-fill so the CNN can learn
    # to discount those timesteps rather than propagating NaN through conv layers.
    df = df.fillna(0.0, subset=sensor_cols)

    log.info("Assigning 5-second windows and filtering complete segments …")
    df_windowed = assign_windows_and_filter(df, sensor_cols)

    write_window_parquet(df_windowed, SEQUENCE_WINDOWS_PARQUET)

    n_complete = (
        spark.read.parquet(SEQUENCE_WINDOWS_PARQUET)
        .select("subject_id", "activity_id", "local_window_id")
        .distinct()
        .count()
    )
    log.info(f"Complete 500-step windows retained: {n_complete:,}")

    spark.stop()
    log.info("Spark stage complete — converting Parquet → NumPy …")

    X, y, meta_df = parquet_to_numpy(SEQUENCE_WINDOWS_PARQUET, sensor_cols)

    np.save(SEQUENCE_TENSOR_NPY, X)
    np.save(SEQUENCE_LABELS_NPY, y)
    meta_df.to_csv(SEQUENCE_META_CSV, index=False)

    log.info(f"Saved tensor   → {SEQUENCE_TENSOR_NPY}   shape={X.shape}")
    log.info(f"Saved labels   → {SEQUENCE_LABELS_NPY}   shape={y.shape}")
    log.info(f"Saved metadata → {SEQUENCE_META_CSV}")

    unique_lbl, counts = np.unique(y, return_counts=True)
    log.info(f"\nClass distribution across {len(unique_lbl)} activity_ids:")
    log.info(f"  {'activity_id':>12}  {'windows':>8}")
    log.info(f"  {'-'*12}  {'-'*8}")
    for lbl, cnt in zip(unique_lbl, counts):
        log.info(f"  {lbl:>12}  {cnt:>8}")

    log.info("\nSequence tensor generation complete.")
    return X, y, meta_df, sensor_cols


if __name__ == "__main__":
    run()
