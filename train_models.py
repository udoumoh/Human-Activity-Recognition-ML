"""
Standalone model training script for PAMAP2 activity classification.
Trains 4 models with CrossValidator, saves results to JSON.
"""

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, isnan, when

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    MultilayerPerceptronClassifier,
    LinearSVC,
    OneVsRest,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import time, json

# ── Spark session ────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("PAMAP2_ModelTraining")
    .master("local[4]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.python.worker.reuse", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(f"Spark version : {spark.version}")

# ── Load features ────────────────────────────────────────────
INPUT_PATH = r"C:/Users/johnu/Desktop/BigDataProject/data/pamap2_features.parquet"
df = spark.read.parquet(INPUT_PATH)
print(f"Loaded {df.count():,} rows x {len(df.columns)} columns")

META = {"subject_id", "activity_id"}
feature_cols = sorted([
    c for c in df.columns
    if c not in META and isinstance(df.schema[c].dataType, DoubleType)
])
print(f"Feature columns: {len(feature_cols)}")

# Replace NaN with 0
for c in feature_cols:
    df = df.withColumn(c, when(isnan(col(c)), 0.0).otherwise(col(c)))
df_clean = df.na.drop(subset=feature_cols)
print(f"After NaN-fix: {df_clean.count():,} rows")

train_df, test_df = df_clean.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
test_df.cache()
print(f"Train: {train_df.count():,}  Test: {test_df.count():,}")

# ── Shared stages ────────────────────────────────────────────
label_indexer = StringIndexer(inputCol="activity_id", outputCol="label").setHandleInvalid("keep")
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw", handleInvalid="keep")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)

eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
eval_f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

results = []

def evaluate_model(name, cv_model, test_data):
    t0 = time.time()
    preds = cv_model.transform(test_data)
    acc = eval_acc.evaluate(preds)
    f1  = eval_f1.evaluate(preds)
    elapsed = time.time() - t0
    results.append({"model": name, "accuracy": round(acc, 4), "f1_weighted": round(f1, 4), "eval_time_s": round(elapsed, 1)})
    print(f"\n{'='*56}\n  {name}\n{'='*56}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Weighted F1 : {f1:.4f}")
    print(f"  Eval time   : {elapsed:.1f}s\n{'='*56}")
    return preds

# ── 1. Logistic Regression ───────────────────────────────────
print("\n>>> Logistic Regression")
lr = LogisticRegression(featuresCol="features", labelCol="label", family="multinomial", maxIter=100, elasticNetParam=0.0)
lr_pipe = Pipeline(stages=[label_indexer, assembler, scaler, lr])
lr_grid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1]).build()
lr_cv = CrossValidator(estimator=lr_pipe, estimatorParamMaps=lr_grid, evaluator=eval_f1, numFolds=3, parallelism=1, seed=42)
t0 = time.time()
lr_model = lr_cv.fit(train_df)
print(f"LR training: {time.time()-t0:.1f}s")
evaluate_model("Logistic Regression", lr_model, test_df)

# ── 2. Random Forest ────────────────────────────────────────
print("\n>>> Random Forest")
rf = RandomForestClassifier(featuresCol="features", labelCol="label", seed=42)
rf_pipe = Pipeline(stages=[label_indexer, assembler, scaler, rf])
rf_grid = ParamGridBuilder().addGrid(rf.numTrees, [20, 50]).addGrid(rf.maxDepth, [5]).build()
rf_cv = CrossValidator(estimator=rf_pipe, estimatorParamMaps=rf_grid, evaluator=eval_f1, numFolds=3, parallelism=1, seed=42)
t0 = time.time()
rf_model = rf_cv.fit(train_df)
print(f"RF training: {time.time()-t0:.1f}s")
evaluate_model("Random Forest", rf_model, test_df)

# ── 3. MLP ──────────────────────────────────────────────────
print("\n>>> Multilayer Perceptron")
NUM_FEATURES = len(feature_cols)
NUM_CLASSES = train_df.select("activity_id").distinct().count()
mlp = MultilayerPerceptronClassifier(featuresCol="features", labelCol="label",
                                      layers=[NUM_FEATURES, 64, NUM_CLASSES], blockSize=128, seed=42)
mlp_pipe = Pipeline(stages=[label_indexer, assembler, scaler, mlp])
mlp_grid = ParamGridBuilder().addGrid(mlp.maxIter, [50, 100]).build()
mlp_cv = CrossValidator(estimator=mlp_pipe, estimatorParamMaps=mlp_grid, evaluator=eval_f1, numFolds=3, parallelism=1, seed=42)
t0 = time.time()
mlp_model = mlp_cv.fit(train_df)
print(f"MLP training: {time.time()-t0:.1f}s")
evaluate_model("Multilayer Perceptron", mlp_model, test_df)

# ── 4. Linear SVM (OVR) ─────────────────────────────────────
print("\n>>> Linear SVM (OneVsRest)")
lsvc = LinearSVC(featuresCol="features", labelCol="label", maxIter=50)
ovr = OneVsRest(classifier=lsvc, featuresCol="features", labelCol="label")
svm_pipe = Pipeline(stages=[label_indexer, assembler, scaler, ovr])
svm_grid = ParamGridBuilder().addGrid(lsvc.regParam, [0.01, 0.1]).build()
svm_cv = CrossValidator(estimator=svm_pipe, estimatorParamMaps=svm_grid, evaluator=eval_f1, numFolds=3, parallelism=1, seed=42)
t0 = time.time()
svm_model = svm_cv.fit(train_df)
print(f"SVM training: {time.time()-t0:.1f}s")
evaluate_model("Linear SVM (OVR)", svm_model, test_df)

# ── Summary ─────────────────────────────────────────────────
print("\n" + "="*64)
print("  MODEL COMPARISON")
print("="*64)
for r in sorted(results, key=lambda x: x["f1_weighted"], reverse=True):
    print(f"  {r['model']:25s}  Acc={r['accuracy']:.4f}  F1={r['f1_weighted']:.4f}")

best = max(results, key=lambda r: r["f1_weighted"])
print(f"\nBest: {best['model']} (F1={best['f1_weighted']})")

# ── Save results ────────────────────────────────────────────
OUTPUT_DIR = r"C:/Users/johnu/Desktop/BigDataProject/data"
with open(f"{OUTPUT_DIR}/model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {OUTPUT_DIR}/model_results.json")

# Save best model
best_models = {"Logistic Regression": lr_model, "Random Forest": rf_model,
               "Multilayer Perceptron": mlp_model, "Linear SVM (OVR)": svm_model}
best_models[best["model"]].bestModel.write().overwrite().save(f"{OUTPUT_DIR}/best_model")
print(f"Best model saved to {OUTPUT_DIR}/best_model")

spark.stop()
print("Done.")
