"""
Weak Scaling Benchmark — Data grows proportionally with cores.

Each core always handles ~25% of the data, so ideal weak scaling
means constant training time regardless of scale.

Configurations:
  local[1]  x  0.25  →  each core handles 0.25
  local[2]  x  0.50  →  each core handles 0.25
  local[4]  x  1.00  →  each core handles 0.25
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, isnan, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import time, gc, json, csv, sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

# -- Weak scaling pairs: (cores, fraction) such that frac/cores is constant --
WEAK_CONFIGS = [
    ("local[1]", 0.25),
    ("local[2]", 0.50),
    ("local[4]", 1.00),
]
CV_FOLDS = 3
SEED = 42

INPUT_PATH = r"C:/Users/johnu/Desktop/BigDataProject/data/pamap2_features.parquet"

print(f"Weak scaling: {len(WEAK_CONFIGS)} configurations (data/cores ratio constant)")
print(f"CV folds per run: {CV_FOLDS}")

bench_results = []

for cores, frac in WEAK_CONFIGS:
    tag = f"cores={cores}, frac={frac}"
    print(f"\n{'-' * 60}")
    print(f"  {tag}  (data per core = {frac/int(cores.split('[')[1].rstrip(']')):.2f})")
    print(f"{'-' * 60}")

    # -- 1. Fresh SparkSession --
    active = SparkSession.getActiveSession()
    if active is not None:
        active.stop()
        gc.collect()
        time.sleep(1)

    spark_bench = (
        SparkSession.builder
        .appName(f"WeakScale_{cores}_{frac}")
        .master(cores)
        .config("spark.driver.memory", "2g")
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:G1HeapRegionSize=16m")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.python.worker.reuse", "true")
        .getOrCreate()
    )
    spark_bench.sparkContext.setLogLevel("ERROR")

    # -- 2. Load & sample --
    df_all = spark_bench.read.parquet(INPUT_PATH)

    META = {"subject_id", "activity_id"}
    feat_cols = sorted([
        c for c in df_all.columns
        if c not in META
        and isinstance(df_all.schema[c].dataType, DoubleType)
    ])

    for c in feat_cols:
        df_all = df_all.withColumn(c, when(isnan(col(c)), 0.0).otherwise(col(c)))
    df_all = df_all.na.drop(subset=feat_cols)

    if frac < 1.0:
        df_sampled = df_all.sample(withReplacement=False, fraction=frac, seed=SEED)
    else:
        df_sampled = df_all

    n_rows = df_sampled.count()
    train_df, test_df = df_sampled.randomSplit([0.8, 0.2], seed=SEED)
    train_df.cache()
    test_df.cache()
    train_count = train_df.count()
    test_count = test_df.count()

    print(f"  Rows: {n_rows:,}  (train {train_count:,} / test {test_count:,})")

    # -- 3. Pipeline (same as strong scaling for fair comparison) --
    idx = StringIndexer(inputCol="activity_id",
                        outputCol="label").setHandleInvalid("keep")
    asm = VectorAssembler(inputCols=feat_cols,
                          outputCol="features_raw",
                          handleInvalid="skip")
    scl = StandardScaler(inputCol="features_raw",
                         outputCol="features",
                         withMean=True, withStd=True)
    rf = RandomForestClassifier(featuresCol="features",
                                labelCol="label",
                                numTrees=20,
                                maxDepth=5,
                                seed=SEED)

    pipe = Pipeline(stages=[idx, asm, scl, rf])

    grid = (ParamGridBuilder()
            .addGrid(rf.numTrees, [20])
            .build())

    cv = CrossValidator(
        estimator=pipe,
        estimatorParamMaps=grid,
        evaluator=MulticlassClassificationEvaluator(
            labelCol="label", metricName="f1"),
        numFolds=CV_FOLDS,
        seed=SEED,
    )

    # -- 4. Train + time --
    t_start = time.time()
    cv_model = cv.fit(train_df)
    train_time = time.time() - t_start

    # -- 5. Evaluate --
    preds = cv_model.transform(test_df)
    acc = MulticlassClassificationEvaluator(
        labelCol="label", metricName="accuracy"
    ).evaluate(preds)
    f1 = MulticlassClassificationEvaluator(
        labelCol="label", metricName="f1"
    ).evaluate(preds)

    n_cores = int(cores.split("[")[1].rstrip("]"))
    bench_results.append({
        "cores": cores,
        "n_cores": n_cores,
        "fraction": frac,
        "rows": n_rows,
        "train_rows": train_count,
        "train_sec": round(train_time, 1),
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1, 4),
    })

    print(f"  Train time : {train_time:.1f}s")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1         : {f1:.4f}")

    # -- 6. Cleanup --
    train_df.unpersist()
    test_df.unpersist()
    spark_bench.stop()
    gc.collect()
    time.sleep(3)

# -- Results summary --
print(f"\n{'=' * 60}")
print(f"  All {len(bench_results)} weak scaling runs complete.")
print(f"{'=' * 60}")

print("\n" + "=" * 80)
print("  WEAK SCALING BENCHMARK -- RF (20 trees, depth 5, 3-fold CV)")
print("  Data per core held constant at ~25% of total dataset")
print("=" * 80)
print(f"  {'Cores':<12} {'Frac':<8} {'Rows':<8} {'Train(s)':<12} {'Acc':<10} {'F1':<10}")
print("-" * 70)
for r in bench_results:
    print(f"  {r['cores']:<12} {r['fraction']:<8} {r['rows']:<8} {r['train_sec']:<12} {r['accuracy']:<10} {r['f1_weighted']:<10}")

# -- Weak scaling efficiency --
base_time = bench_results[0]["train_sec"]
print("\n" + "=" * 80)
print("  WEAK SCALING EFFICIENCY  (ideal = 1.0)")
print("  Efficiency = T_base / T_n  where T_base is time at 1 core / 25% data")
print("=" * 80)
print(f"  {'Cores':<12} {'Frac':<8} {'Train(s)':<12} {'Efficiency':<12}")
print("-" * 50)
for r in bench_results:
    eff = round(base_time / r["train_sec"], 2) if r["train_sec"] > 0 else 0.0
    print(f"  {r['cores']:<12} {r['fraction']:<8} {r['train_sec']:<12} {eff:<12}")

# -- Save results --
OUTPUT_DIR = r"C:/Users/johnu/Desktop/BigDataProject/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

json_path = f"{OUTPUT_DIR}/weak_scaling.json"
with open(json_path, "w") as f:
    json.dump(bench_results, f, indent=2)
print(f"\nSaved JSON to {json_path}")

csv_path = f"{OUTPUT_DIR}/weak_scaling.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=bench_results[0].keys())
    writer.writeheader()
    writer.writerows(bench_results)
print(f"Saved CSV  to {csv_path}")

print("Done.")
