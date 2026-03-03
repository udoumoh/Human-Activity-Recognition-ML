# PAMAP2 Activity Recognition — PySpark ML Pipeline

## Medallion Architecture

This project implements a **Medallion Architecture** (Bronze → Silver → Gold) for processing the PAMAP2 Physical Activity Monitoring dataset using PySpark. The architecture separates data processing into three distinct layers, each with a clear responsibility.

```
project/
├── bronze/                          # Raw data ingestion
│   ├── ingestion.py                 # Load .dat files → Parquet
│   └── raw_data_schema.py           # 54-column PAMAP2 schema
│
├── silver/                          # Cleaning & preprocessing
│   ├── cleaning.py                  # Remove invalid rows/columns
│   └── preprocessing.py             # HR interpolation, normalisation
│
├── gold/                            # Analytics & ML
│   ├── feature_engineering.py       # Sliding-window aggregation
│   ├── model_training.py            # MLlib model training (4 models)
│   ├── evaluation.py                # Confusion matrix, metrics, charts
│   └── tableau_export.py            # Aggregate CSVs for dashboards
│
├── config/
│   └── settings.py                  # Centralised paths & constants
│
├── src/
│   └── window_transformer.py        # Custom PySpark ML Transformer
│
├── notebooks/                       # Interactive exploration
│   ├── 01_ingestion.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_evaluation_scaling.ipynb
│   ├── 06_sklearn_baseline.ipynb
│   ├── 07_model_evaluation.ipynb
│   └── 08_stability_analysis.ipynb
│
├── data/
│   ├── bronze/                      # Raw Parquet (from ingestion)
│   ├── silver/                      # Cleaned Parquet
│   └── gold/                        # Features + trained models
│
└── results/
    ├── tableau_exports/             # Tableau-ready CSVs
    └── *.png / *.csv                # Charts and metric tables
```

## Layer Descriptions

### Bronze Layer — Raw Data Ingestion

The Bronze layer preserves **raw, unprocessed data** exactly as it arrives from the source (PAMAP2 `.dat` files from the UCI repository). No cleaning, imputation, or transformation is applied. This ensures full data lineage traceability — if cleaning logic changes, only the Silver layer needs to be re-run.

**Key operations:**
- Distributed CSV parsing via `spark.read.csv()` with explicit 54-column schema
- Subject ID extraction from file paths using native SQL expressions
- Output partitioned by `subject_id` for downstream partition pruning
- Basic validation (null checks, row counts) without modifying data

**Run:** `python -m bronze.ingestion`

### Silver Layer — Cleaning & Preprocessing

The Silver layer transforms Bronze data into a **validated, quality-assured** dataset. It is split into two stages for modularity:

1. **Cleaning** (`cleaning.py`): Structural fixes — removes transient activity rows (ID=0), drops 12 invalid orientation columns, filters complete sensor dropouts.
2. **Preprocessing** (`preprocessing.py`): Value-level transforms — heart-rate interpolation via bounded Window functions, min-max normalisation using a single-pass distributed aggregation.

**Key design decisions:**
- DataFrame API over RDD: Catalyst optimizer fuses filter/drop/transform operations into a single physical plan stage, reducing serialisation overhead.
- Bounded window functions (15 rows) for HR interpolation instead of unbounded windows, limiting scan to O(1) per row after the initial sort.
- NaN-safe normalisation using `when(~isnan())` to prevent NaN propagation into min/max statistics.

**Run:** `python -m silver.cleaning && python -m silver.preprocessing`

### Gold Layer — Feature Engineering, ML & Analytics

The Gold layer produces **business-level, analytics-ready** outputs:

1. **Feature Engineering** (`feature_engineering.py`): Converts 2.7M time-series rows into 5,447 windowed feature rows (172 features per 5-second window) using a single distributed `groupBy().agg()` pass with 173 aggregation expressions.
2. **Model Training** (`model_training.py`): Trains 4 MLlib classifiers (LR, RF, MLP, SVM) with 3-fold CrossValidator. Best model (MLP, F1=0.896) saved as PipelineModel.
3. **Evaluation** (`evaluation.py`): Confusion matrix, per-class metrics, feature importance via RF surrogate. Distributed prediction + local visualisation.
4. **Tableau Export** (`tableau_export.py`): Consolidates all results into 8 flat CSVs optimised for Tableau dashboard import.

**Run:** `python -m gold.feature_engineering && python -m gold.model_training && python -m gold.evaluation && python -m gold.tableau_export`

## Alignment with Big Data Best Practices

The Medallion Architecture follows established Big Data engineering principles. Each layer has a single responsibility — ingestion, cleaning, or analytics — enabling independent re-execution without cascading side effects. Data flows forward through typed Parquet files, which provide columnar compression, predicate pushdown, and schema enforcement at no additional development cost. The use of PySpark's DataFrame API throughout (rather than RDD or pandas) ensures that all operations — from CSV parsing to cross-validated model training — are distributed across available cores and can scale to cluster deployments without code changes.

## Scalability Design

This architecture is designed to scale from a single laptop to a multi-node cluster. At each layer, data is partitioned by `subject_id`, enabling partition pruning and parallel processing per subject. The Silver layer's bounded window functions and the Gold layer's single-pass aggregation strategy minimise data shuffling — the most expensive operation in distributed computing. Spark's lazy evaluation and Catalyst query optimisation fuse multiple transformations into minimal physical stages, reducing I/O passes. The pipeline has been benchmarked across 1–4 cores with strong scaling speedup of up to 1.31x at 4 cores, and weak scaling efficiency above 1.0 (super-linear), confirming that the architecture handles proportionally larger workloads efficiently as resources increase.

## Dataset

**PAMAP2 Physical Activity Monitoring Dataset** (UCI Machine Learning Repository)
- 9 subjects, 18 activities, 3 IMU placements (hand, chest, ankle)
- 54 raw columns, 3.85M rows, 1.61 GB
- 100 Hz IMU sampling rate, ~9 Hz heart rate

## Models

| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| Multilayer Perceptron | 89.65% | 0.8961 |
| Logistic Regression | 87.92% | 0.8749 |
| Linear SVM (OVR) | 85.81% | 0.8484 |
| Random Forest | 75.36% | 0.7195 |
