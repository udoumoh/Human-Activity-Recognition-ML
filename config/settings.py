"""
Centralised configuration for the PAMAP2 Medallion Architecture pipeline.

All paths, schema parameters, and processing constants are defined here
so that Bronze, Silver, and Gold layers share a single source of truth.
"""

import os

PROJECT_ROOT = r"C:/Users/johnu/Desktop/BigDataProject"

DATA_ROOT = r"C:/Users/johnu/Downloads/pamap2+physical+activity+monitoring/PAMAP2_Dataset"
PROTOCOL_PATH = os.path.join(DATA_ROOT, "Protocol")
OPTIONAL_PATH = os.path.join(DATA_ROOT, "Optional")

BRONZE_OUTPUT = os.path.join(PROJECT_ROOT, "data", "bronze", "pamap2_raw.parquet")
SILVER_OUTPUT = os.path.join(PROJECT_ROOT, "data", "pamap2_clean.parquet")
GOLD_FEATURES_OUTPUT = os.path.join(PROJECT_ROOT, "data", "pamap2_features.parquet")
GOLD_MODEL_OUTPUT = os.path.join(PROJECT_ROOT, "data", "best_model")
GOLD_RESULTS_OUTPUT = os.path.join(PROJECT_ROOT, "data", "model_results.json")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TABLEAU_EXPORT_DIR = os.path.join(RESULTS_DIR, "tableau_exports")

# Intermediate Parquet written by Spark (avoids driver collect OOM)
SEQUENCE_WINDOWS_PARQUET = os.path.join(PROJECT_ROOT, "data", "gold", "sequence_windows.parquet")
# Final NumPy tensors (post-Spark, driver-side reshape)
SEQUENCE_TENSOR_NPY  = os.path.join(PROJECT_ROOT, "data", "gold", "sequence_tensor.npy")
SEQUENCE_LABELS_NPY  = os.path.join(PROJECT_ROOT, "data", "gold", "sequence_labels.npy")
SEQUENCE_META_CSV    = os.path.join(PROJECT_ROOT, "data", "gold", "sequence_meta.csv")
CNN_METRICS_CSV      = os.path.join(PROJECT_ROOT, "results", "cnn_metrics.csv")
CNN_BENCHMARK_CSV    = os.path.join(PROJECT_ROOT, "results", "cnn_benchmark.csv")
MODEL_COMPARISON_CSV = os.path.join(PROJECT_ROOT, "results", "model_comparison.csv")
LEARNING_CURVES_DIR  = os.path.join(PROJECT_ROOT, "results", "learning_curves")

CNN_BATCH_SIZE = 64
CNN_EPOCHS     = 100
CNN_LR         = 1e-3
CNN_PATIENCE   = 10      # early-stopping patience (epochs)
CNN_DROPOUT    = 0.3

# Tuned for local mode on 14 GB RAM
SPARK_DRIVER_MEMORY = "4g"
SPARK_SHUFFLE_PARTITIONS = "8"

SAMPLE_RATE_HZ = 100
IMU_LOCATIONS = ["hand", "chest", "ankle"]
IMU_SUFFIXES = [
    "temperature",
    "acc_16g_x", "acc_16g_y", "acc_16g_z",
    "acc_6g_x",  "acc_6g_y",  "acc_6g_z",
    "gyro_x",    "gyro_y",    "gyro_z",
    "mag_x",     "mag_y",     "mag_z",
    "orient_0",  "orient_1",  "orient_2", "orient_3",
]

WINDOW_DURATION_SEC = 5.0
MIN_WINDOW_FILL = 0.5   # reject windows with < 50% of expected samples

META_COLS = {"timestamp", "activity_id", "subject_id", "session_type"}

# HR gap at 100 Hz IMU / 9 Hz HR is ~11 rows; 15-row window covers that with margin
HR_FILL_WINDOW = 15

TRAIN_TEST_SPLIT = [0.8, 0.2]
RANDOM_SEED = 42
CV_FOLDS = 3
