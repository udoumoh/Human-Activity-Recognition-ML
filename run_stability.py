"""
Stability Analysis — Run the best model (MLP) 5 times with different
random seeds to measure variance in accuracy and F1.

Each run uses a different train/test split seed while keeping the
model architecture and hyperparameters fixed (best config from CV).
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, isnan, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import time, gc, json, csv, sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

# -- Configuration --
SEEDS = [42, 123, 256, 789, 1024]
INPUT_PATH = r"C:/Users/johnu/Desktop/BigDataProject/data/pamap2_features.parquet"

print(f"Stability Analysis: MLP trained {len(SEEDS)} times with different seeds")
print(f"Seeds: {SEEDS}")

results = []

for i, seed in enumerate(SEEDS):
    print(f"\n{'-' * 60}")
    print(f"  Run {i+1}/{len(SEEDS)}  seed={seed}")
    print(f"{'-' * 60}")

    # -- Fresh SparkSession --
    active = SparkSession.getActiveSession()
    if active is not None:
        active.stop()
        gc.collect()
        time.sleep(1)

    spark = (
        SparkSession.builder
        .appName(f"Stability_seed{seed}")
        .master("local[4]")
        .config("spark.driver.memory", "3g")
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:G1HeapRegionSize=16m")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.python.worker.reuse", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # -- Load data --
    df = spark.read.parquet(INPUT_PATH)

    META = {"subject_id", "activity_id"}
    feature_cols = sorted([
        c for c in df.columns
        if c not in META
        and isinstance(df.schema[c].dataType, DoubleType)
    ])

    for c in feature_cols:
        df = df.withColumn(c, when(isnan(col(c)), 0.0).otherwise(col(c)))
    df = df.na.drop(subset=feature_cols)

    # -- Different split per seed --
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=seed)
    train_df.cache()
    test_df.cache()

    num_features = len(feature_cols)
    num_classes = train_df.select("activity_id").distinct().count()
    train_count = train_df.count()
    test_count = test_df.count()

    print(f"  Train: {train_count:,}  Test: {test_count:,}")
    print(f"  Features: {num_features}  Classes: {num_classes}")

    # -- Pipeline with fixed best hyperparameters (maxIter=100) --
    idx = StringIndexer(inputCol="activity_id", outputCol="label").setHandleInvalid("keep")
    asm = VectorAssembler(inputCols=feature_cols, outputCol="features_raw", handleInvalid="keep")
    scl = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    mlp = MultilayerPerceptronClassifier(
        featuresCol="features",
        labelCol="label",
        layers=[num_features, 64, num_classes],
        blockSize=128,
        maxIter=100,
        seed=seed,
    )

    pipe = Pipeline(stages=[idx, asm, scl, mlp])

    # -- Train --
    t0 = time.time()
    model = pipe.fit(train_df)
    train_time = time.time() - t0

    # -- Evaluate --
    preds = model.transform(test_df)

    eval_acc = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    eval_f1 = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")
    eval_prec = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedPrecision")
    eval_rec = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedRecall")

    acc = eval_acc.evaluate(preds)
    f1 = eval_f1.evaluate(preds)
    prec = eval_prec.evaluate(preds)
    rec = eval_rec.evaluate(preds)

    results.append({
        "run": i + 1,
        "seed": seed,
        "train_rows": train_count,
        "test_rows": test_count,
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted": round(rec, 4),
        "train_sec": round(train_time, 1),
    })

    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1         : {f1:.4f}")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  Train time : {train_time:.1f}s")

    # -- Cleanup --
    train_df.unpersist()
    test_df.unpersist()
    spark.stop()
    gc.collect()
    time.sleep(2)

# -- Summary statistics --
import statistics

accs = [r["accuracy"] for r in results]
f1s = [r["f1_weighted"] for r in results]

print(f"\n{'=' * 60}")
print(f"  STABILITY ANALYSIS SUMMARY  ({len(results)} runs)")
print(f"{'=' * 60}")
print(f"  {'Run':<6} {'Seed':<8} {'Accuracy':<12} {'F1':<12} {'Train(s)':<10}")
print(f"  {'-'*48}")
for r in results:
    print(f"  {r['run']:<6} {r['seed']:<8} {r['accuracy']:<12} {r['f1_weighted']:<12} {r['train_sec']:<10}")

print(f"\n  Accuracy  : mean={statistics.mean(accs):.4f}  std={statistics.stdev(accs):.4f}  "
      f"range=[{min(accs):.4f}, {max(accs):.4f}]")
print(f"  F1        : mean={statistics.mean(f1s):.4f}  std={statistics.stdev(f1s):.4f}  "
      f"range=[{min(f1s):.4f}, {max(f1s):.4f}]")

# -- Save results --
OUTPUT_DIR = r"C:/Users/johnu/Desktop/BigDataProject/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_path = f"{OUTPUT_DIR}/stability_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\nSaved CSV to {csv_path}")

json_path = f"{OUTPUT_DIR}/stability_results.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved JSON to {json_path}")

print("Done.")
