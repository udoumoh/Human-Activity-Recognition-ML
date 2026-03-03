"""
Gold Layer — Raw Sequence Tensor Generation (Hybrid Pipeline, Stage 1/2).

PURPOSE
-------
Transforms the Silver layer (2.7 M rows of 100 Hz sensor data) into a
3-D NumPy tensor suitable for the 1-D CNN:

    tensor shape : (N_windows, 500, N_channels)
    label shape  : (N_windows,)              [activity_id per window]
    meta CSV     : window_id, subject_id, activity_id

This is architecturally distinct from the Gold statistical-feature path
(gold/feature_engineering.py).  That path compresses each 5-second window
into 172 scalar statistics; this path retains the *full time series* so
that the CNN can learn temporal patterns from raw inertial signals.

WINDOW STRATEGY
---------------
- Duration  : 5 seconds  (WINDOW_DURATION_SEC = 5.0)
- Timesteps : 500 samples (SAMPLE_RATE_HZ × WINDOW_DURATION_SEC)
- Grouping  : Per (subject_id, activity_id) — windows never straddle
              two activities; avoids label contamination at boundaries.
- Filter    : Only COMPLETE windows (exactly 500 timesteps) are retained.
              Boundary-incomplete windows are discarded.
- Window ID : floor((timestamp − t_min) / 5.0)  — same formula as
              feature_engineering.py, ensuring consistent segmentation.

MEMORY JUSTIFICATION
--------------------
A naive driver-side collect() of 2.7 M rows × 40 channels × 8 bytes
would require ~864 MB of driver heap.  Instead:

  Step A (Spark)  — window assignment, filtering, Parquet write
                    stays entirely on executors / disk.
  Step B (driver) — PyArrow reads the compact windowed Parquet
                    (~150 MB compressed) and reshapes into the
                    final tensor (424 MB float32).
  Peak driver RSS ≈ 574 MB — well within the 4 GB driver allocation.

SHUFFLE JUSTIFICATION
---------------------
Exactly TWO shuffle stages are introduced:
  1. Window.partitionBy("subject_id","activity_id")
       Groups all rows for a (subject, activity) pair onto the same set
       of executors.  Max partitions: 9 subjects × 18 activities = 162.
  2. groupBy(...).count() for window completeness check.
       Because Step 1 already co-located rows by group key, Catalyst
       can use a map-side partial aggregation (no cross-node shuffle for
       the count query if AQE is enabled).
  3. join(win_counts) uses a BroadcastHashJoin (win_counts ≈ 130 KB)
       — zero additional shuffle.

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

# ── Project imports ──────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import (
    SILVER_OUTPUT,
    SEQUENCE_WINDOWS_PARQUET, SEQUENCE_TENSOR_NPY,
    SEQUENCE_LABELS_NPY, SEQUENCE_META_CSV,
    SPARK_DRIVER_MEMORY, SPARK_SHUFFLE_PARTITIONS,
    WINDOW_DURATION_SEC, SAMPLE_RATE_HZ, META_COLS,
)

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SEQ-TENSOR] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Derived constants ────────────────────────────────────────
WINDOW_STEPS = int(SAMPLE_RATE_HZ * WINDOW_DURATION_SEC)   # 500 timesteps
# We require full 500-step windows only (no boundary-partial windows)
REQUIRED_STEPS = WINDOW_STEPS


# ─────────────────────────────────────────────────────────────
# 1. Spark session
# ─────────────────────────────────────────────────────────────

def get_spark() -> SparkSession:
    """
    SparkSession tuned for the sequence tensor generation job.

    local[4]  — uses 4 CPU threads; the bottleneck for this job is
                 the I/O-bound Silver read and the shuffle for grouping,
                 so extra threads help throughput without contending for
                 driver memory.
    AQE enabled — allows Spark to coalesce shuffle partitions after the
                   groupBy count, avoiding many tiny partitions.
    """
    return (
        SparkSession.builder
        .appName("HAR-SequenceTensor")
        .master("local[4]")
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # Broadcast threshold: 10 MB covers the completeness-filter table
        # (max ~5 K rows × 3 cols × 8 bytes ≈ 120 KB)
        .config("spark.sql.autoBroadcastJoinThreshold", str(10 * 1024 * 1024))
        .getOrCreate()
    )


# ─────────────────────────────────────────────────────────────
# 2. Sensor column detection
# ─────────────────────────────────────────────────────────────

def identify_sensor_cols(df) -> list:
    """
    Return sorted list of DoubleType columns that are not metadata.

    Excludes META_COLS = {timestamp, activity_id, subject_id, session_type}
    and any StringType columns (e.g., activity_name added by cleaning.py).

    On the PAMAP2 Silver schema this resolves to ~40 channels:
        heart_rate (1) + hand_* (13) + chest_* (13) + ankle_* (13) = 40
    (orientation columns were removed in silver/cleaning.py)
    """
    exclude = set(META_COLS) | {"activity_name"}
    return sorted([
        f.name for f in df.schema.fields
        if isinstance(f.dataType, DoubleType) and f.name not in exclude
    ])


# ─────────────────────────────────────────────────────────────
# 3. Window assignment and completeness filter (Spark)
# ─────────────────────────────────────────────────────────────

def assign_windows_and_filter(df, sensor_cols: list):
    """
    Assign 5-second window IDs and retain only complete (500-step) windows.

    Returns a Spark DataFrame with columns:
        subject_id, activity_id, local_window_id, timestamp, *sensor_cols

    Algorithm
    ---------
    1. Partition by (subject_id, activity_id) — keeps group data co-located.
    2. Compute t_min per group via Window function (no additional shuffle).
    3. local_window_id = floor((t − t_min) / WINDOW_DURATION_SEC)
       — matches formula in gold/feature_engineering.py exactly.
    4. Count rows per (subject, activity, local_window_id).
       Filter where count == 500 (complete windows only).
    5. Broadcast-join the completeness table back to filter the main DF.
       The join key table is ~130 KB → automatically broadcast (no shuffle).
    """
    group_keys = ["subject_id", "activity_id"]

    # Step 1-2: group-local Window — ONE shuffle to co-locate rows
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

    # Step 4: completeness filter table (tiny — will be auto-broadcast)
    win_counts = (
        df
        .groupBy(*group_keys, "local_window_id")
        .agg(F.count("*").alias("_n_steps"))
        .filter(F.col("_n_steps") == REQUIRED_STEPS)
        .drop("_n_steps")
        # Explicit broadcast hint documents intent; Spark will broadcast
        # automatically given autoBroadcastJoinThreshold = 10 MB
        .hint("broadcast")
    )

    # Step 5: BroadcastHashJoin (zero shuffle)
    df_complete = df.join(
        win_counts,
        on=[*group_keys, "local_window_id"],
        how="inner",
    )

    # Select only the columns needed downstream
    select_cols = [*group_keys, "local_window_id", "timestamp"] + sensor_cols
    return df_complete.select(*select_cols)


# ─────────────────────────────────────────────────────────────
# 4. Distributed Parquet write
# ─────────────────────────────────────────────────────────────

def write_window_parquet(df, path: str) -> None:
    """
    Write windowed rows to Parquet partitioned by subject_id (9 files).

    Partitioning by subject_id (not by window_id) keeps partition count
    low (9) and avoids the "many small files" problem.  Within each
    partition file rows are physically sorted by (local_window_id, timestamp)
    so that the driver-side NumPy reshape can simply iterate sequentially.

    Compression: snappy (default).  Expected on-disk size: ~100-150 MB
    (down from ~465 MB uncompressed) due to repeated sensor patterns.
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


# ─────────────────────────────────────────────────────────────
# 5. Parquet → NumPy reshape (driver-side, post-Spark)
# ─────────────────────────────────────────────────────────────

def parquet_to_numpy(parquet_path: str, sensor_cols: list):
    """
    Read the windowed Parquet and reshape into a 3-D tensor.

    Memory path
    -----------
    PyArrow reads the Parquet directory in column-major chunks
    (no full in-memory copy until .to_pandas()).  Peak driver RSS:
        Pandas DataFrame : ~465 MB  (2.7 M rows × 44 cols × 4 bytes)
        NumPy tensor     : ~424 MB  (N × 500 × 40 × float32)
        Total peak       : ~580 MB  (DataFrame freed before full tensor built)

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

    # Sort globally by (subject, activity, local_window_id, timestamp)
    # within-partition sort from Spark covers most of this; the global
    # sort here reconciles across the subject_id partition files.
    df = df.sort_values(
        ["subject_id", "activity_id", "local_window_id", "timestamp"]
    ).reset_index(drop=True)

    log.info(f"Pandas read done in {time.time() - t0:.1f}s  "
             f"({len(df):,} rows, {df.shape[1]} cols)")

    # Identify unique windows
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

    # Vectorised fill: use numpy grouping via searchsorted on sorted win_idx
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


# ─────────────────────────────────────────────────────────────
# 6. Main orchestrator
# ─────────────────────────────────────────────────────────────

def run():
    """End-to-end: Silver → sequence_tensor.npy + sequence_labels.npy."""
    log.info("=" * 65)
    log.info("SEQUENCE TENSOR GENERATION  (Silver → Gold/DL)")
    log.info("=" * 65)

    # Ensure output directories exist
    os.makedirs(os.path.join(PROJECT_ROOT, "data", "gold"), exist_ok=True)
    os.makedirs(PROJECT_ROOT, exist_ok=True)

    # ── A: Spark stage ───────────────────────────────────────
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    log.info(f"Loading Silver from {SILVER_OUTPUT} …")
    df = spark.read.parquet(SILVER_OUTPUT)
    log.info(f"Silver rows: {df.count():,}  |  columns: {len(df.columns)}")

    # Remove transient rows (activity_id == 0 means no label)
    df = df.filter(F.col("activity_id") != 0)

    # Detect sensor channels
    sensor_cols = identify_sensor_cols(df)
    log.info(f"Sensor channels: {len(sensor_cols)}  "
             f"(e.g. {sensor_cols[:3]} … {sensor_cols[-1]})")

    # Fill null sensor readings with 0 (heart_rate has ~91% nulls at 100 Hz)
    # The CNN will learn to discount zero-padded heart_rate periods
    df = df.fillna(0.0, subset=sensor_cols)

    # Assign windows and filter for completeness
    log.info("Assigning 5-second windows and filtering complete segments …")
    df_windowed = assign_windows_and_filter(df, sensor_cols)

    # Write to Parquet (distributed, no driver collect)
    write_window_parquet(df_windowed, SEQUENCE_WINDOWS_PARQUET)

    # Count windows for logging (cheap operation on small Parquet)
    n_complete = (
        spark.read.parquet(SEQUENCE_WINDOWS_PARQUET)
        .select("subject_id", "activity_id", "local_window_id")
        .distinct()
        .count()
    )
    log.info(f"Complete 500-step windows retained: {n_complete:,}")

    spark.stop()
    log.info("Spark stage complete — converting Parquet → NumPy …")

    # ── B: Driver-side NumPy reshape ─────────────────────────
    X, y, meta_df = parquet_to_numpy(SEQUENCE_WINDOWS_PARQUET, sensor_cols)

    # ── C: Save outputs ──────────────────────────────────────
    np.save(SEQUENCE_TENSOR_NPY, X)
    np.save(SEQUENCE_LABELS_NPY, y)
    meta_df.to_csv(SEQUENCE_META_CSV, index=False)

    log.info(f"Saved tensor   → {SEQUENCE_TENSOR_NPY}   shape={X.shape}")
    log.info(f"Saved labels   → {SEQUENCE_LABELS_NPY}   shape={y.shape}")
    log.info(f"Saved metadata → {SEQUENCE_META_CSV}")

    # ── D: Class distribution summary ───────────────────────
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
