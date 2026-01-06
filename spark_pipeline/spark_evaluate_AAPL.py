import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# =====================================================
# SPARK SESSION
# =====================================================
spark = (
    SparkSession.builder
    .appName("Spark_Evaluate_AAPL")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "data_full.csv")
MODEL_DIR = os.path.join(BASE_DIR, "spark_models")

# =====================================================
# LOAD DATA
# =====================================================
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

# Ensure correct types
df = df.withColumn("Target", col("Target").cast(DoubleType()))

# Train-test split (same logic as training)
train_df, test_df = df.randomSplit([0.9, 0.1], seed=42)

print(f"Test samples count: {test_df.count()}")

# =====================================================
# LOAD SPARK MODELS (AAPL)
# =====================================================
models = {
    "Logistic Regression": PipelineModel.load(os.path.join(MODEL_DIR, "lr_AAPL")),
    "Random Forest": PipelineModel.load(os.path.join(MODEL_DIR, "rf_AAPL")),
    "Gradient Boosted Trees": PipelineModel.load(os.path.join(MODEL_DIR, "gbt_AAPL"))
}

# =====================================================
# EVALUATORS
# =====================================================
evaluator_accuracy = MulticlassClassificationEvaluator(
    labelCol="Target", predictionCol="prediction", metricName="accuracy"
)

evaluator_precision = MulticlassClassificationEvaluator(
    labelCol="Target", predictionCol="prediction", metricName="weightedPrecision"
)

evaluator_recall = MulticlassClassificationEvaluator(
    labelCol="Target", predictionCol="prediction", metricName="weightedRecall"
)

evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="Target", predictionCol="prediction", metricName="f1"
)

# =====================================================
# EVALUATION LOOP
# =====================================================
results = []

for model_name, model in models.items():
    preds = model.transform(test_df)

    acc = evaluator_accuracy.evaluate(preds)
    prec = evaluator_precision.evaluate(preds)
    rec = evaluator_recall.evaluate(preds)
    f1 = evaluator_f1.evaluate(preds)

    results.append((model_name, acc, prec, rec, f1))

# =====================================================
# DISPLAY RESULTS (TABLE 4.1)
# =====================================================
print("\n Prediction Performance on AAPL Dataset\n")
print("{:<25} {:<10} {:<10} {:<10} {:<10}".format(
    "Model", "Accuracy", "Precision", "Recall", "F1-Score"
))

for r in results:
    print("{:<25} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
        r[0], r[1], r[2], r[3], r[4]
    ))

spark.stop()
