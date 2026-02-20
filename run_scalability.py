"""
Scalability Benchmark — Cores × Sample Fraction
Measures wall-clock training time of a Random Forest pipeline
while varying Spark parallelism and data fraction.
"""

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, isnan, when, round as spark_round
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import time, itertools, gc, json, csv, sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

# ── Experiment grid ──────────────────────────────────────────
CORE_CONFIGS = ["local[1]", "local[2]", "local[4]"]
SAMPLE_FRACS = [0.25, 0.50, 0.75, 1.0]
CV_FOLDS = 3
SEED = 42

INPUT_PATH = r"C:/Users/johnu/Desktop/BigDataProject/data/pamap2_features.parquet"

print(f"Configurations : {len(CORE_CONFIGS)} core settings x {len(SAMPLE_FRACS)} fractions "
      f"= {len(CORE_CONFIGS) * len(SAMPLE_FRACS)} runs")
print(f"CV folds per run: {CV_FOLDS}")

bench_results = []

for cores in CORE_CONFIGS:
    for frac in SAMPLE_FRACS:
        tag = f"cores={cores}, frac={frac}"
        print(f"\n{'-' * 60}")
        print(f"  {tag}")
        print(f"{'-' * 60}")

        # ── 1. Fresh SparkSession ────────────────────────────
        active = SparkSession.getActiveSession()
        if active is not None:
            active.stop()
            gc.collect()
            time.sleep(1)

        spark_bench = (
            SparkSession.builder
            .appName(f"Bench_{cores}_{frac}")
            .master(cores)
            .config("spark.driver.memory", "2g")
            .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:G1HeapRegionSize=16m")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.ui.enabled", "false")
            .config("spark.python.worker.reuse", "true")
            .getOrCreate()
        )
        spark_bench.sparkContext.setLogLevel("ERROR")

        # ── 2. Load & sample ─────────────────────────────────
        df_all = spark_bench.read.parquet(INPUT_PATH)

        META = {"subject_id", "activity_id"}
        feat_cols = sorted([
            c for c in df_all.columns
            if c not in META
            and isinstance(df_all.schema[c].dataType, DoubleType)
        ])

        # Replace NaN with 0
        for c in feat_cols:
            df_all = df_all.withColumn(c, when(isnan(col(c)), 0.0).otherwise(col(c)))
        df_all = df_all.na.drop(subset=feat_cols)

        if frac < 1.0:
            df_sampled = df_all.sample(withReplacement=False,
                                       fraction=frac, seed=SEED)
        else:
            df_sampled = df_all

        n_rows = df_sampled.count()
        train_df, test_df = df_sampled.randomSplit([0.8, 0.2], seed=SEED)
        train_df.cache()
        test_df.cache()
        train_count = train_df.count()
        test_count = test_df.count()

        print(f"  Rows: {n_rows:,}  (train {train_count:,} / test {test_count:,})")

        # ── 3. Pipeline ──────────────────────────────────────
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

        # ── 4. Train + time ──────────────────────────────────
        t_start = time.time()
        cv_model = cv.fit(train_df)
        train_time = time.time() - t_start

        # ── 5. Evaluate ──────────────────────────────────────
        preds = cv_model.transform(test_df)
        acc = MulticlassClassificationEvaluator(
            labelCol="label", metricName="accuracy"
        ).evaluate(preds)
        f1 = MulticlassClassificationEvaluator(
            labelCol="label", metricName="f1"
        ).evaluate(preds)

        bench_results.append({
            "cores": cores,
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

        # ── 6. Cleanup ───────────────────────────────────────
        train_df.unpersist()
        test_df.unpersist()
        spark_bench.stop()
        gc.collect()
        time.sleep(3)

print(f"\n{'=' * 60}")
print(f"  All {len(bench_results)} benchmark runs complete.")
print(f"{'=' * 60}")

# ── Results summary ──────────────────────────────────────────
print("\n" + "=" * 80)
print("  SCALABILITY BENCHMARK -- Random Forest (20 trees, depth 5, 3-fold CV)")
print("=" * 80)
print(f"  {'Cores':<12} {'Frac':<8} {'Rows':<8} {'Train(s)':<12} {'Acc':<10} {'F1':<10}")
print("-" * 70)
for r in sorted(bench_results, key=lambda x: (x["cores"], x["fraction"])):
    print(f"  {r['cores']:<12} {r['fraction']:<8} {r['rows']:<8} {r['train_sec']:<12} {r['accuracy']:<10} {r['f1_weighted']:<10}")

# ── Speedup table ────────────────────────────────────────────
print("\n" + "=" * 80)
print("  SPEEDUP vs local[1]  (time_local1 / time_localN)")
print("=" * 80)

baseline = {r["fraction"]: r["train_sec"] for r in bench_results
            if r["cores"] == "local[1]"}

print(f"  {'Cores':<12} {'Frac':<8} {'Train(s)':<12} {'Speedup':<10}")
print("-" * 50)
for r in sorted(bench_results, key=lambda x: (x["fraction"], x["cores"])):
    base_t = baseline.get(r["fraction"], r["train_sec"])
    speedup = round(base_t / r["train_sec"], 2) if r["train_sec"] > 0 else 0.0
    print(f"  {r['cores']:<12} {r['fraction']:<8} {r['train_sec']:<12} {speedup:<10}")

# ── Save results ─────────────────────────────────────────────
OUTPUT_DIR = r"C:/Users/johnu/Desktop/BigDataProject/data"

out_path = f"{OUTPUT_DIR}/scalability_results.json"
with open(out_path, "w") as f:
    json.dump(bench_results, f, indent=2)
print(f"\nSaved {len(bench_results)} benchmark records to {out_path}")

csv_path = f"{OUTPUT_DIR}/scalability_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=bench_results[0].keys())
    writer.writeheader()
    writer.writerows(bench_results)
print(f"Saved CSV to {csv_path}")

print("Done.")
