import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.pipeline import PipelineModel

# =========================================
# PATH SETUP
# =========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "spark_models")

# =========================================
# SPARK SESSION
# =========================================
spark = (
    SparkSession.builder
    .appName("Spark_Streaming_Predict_AAPL")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("[INFO] Loading AAPL models...")

lr_model  = PipelineModel.load(os.path.join(MODEL_DIR, "lr_AAPL"))
rf_model  = PipelineModel.load(os.path.join(MODEL_DIR, "rf_AAPL"))
gbt_model = PipelineModel.load(os.path.join(MODEL_DIR, "gbt_AAPL"))

print("[OK] AAPL models loaded")

# =========================================
# SCHEMA
# =========================================
schema = StructType([
    StructField("Date", StringType()),
    StructField("Open", DoubleType()),
    StructField("High", DoubleType()),
    StructField("Low", DoubleType()),
    StructField("Close", DoubleType()),
    StructField("Volume", DoubleType()),
    StructField("SMA5", DoubleType()),
    StructField("SMA10", DoubleType()),
    StructField("Ret1", DoubleType()),
    StructField("Ret5", DoubleType()),
    StructField("Target", IntegerType())
])

# =========================================
# READ FROM KAFKA
# =========================================
raw_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "aapl_stream")
    .option("startingOffsets", "latest")
    .load()
)

df = (
    raw_df
    .selectExpr("CAST(value AS STRING)")
    .select(from_json(col("value"), schema).alias("data"))
    .select("data.*")
    .withColumn("ingest_time", current_timestamp())
)

# =========================================
# FOREACH BATCH (SAFE ML INFERENCE)
# =========================================
def process_batch(batch_df, batch_id):
    if batch_df.count() == 0:
        return

    base_df = batch_df.cache()

    # Apply models independently
    lr_out  = lr_model.transform(base_df).select("Date", "Target", col("prediction").alias("lr_pred"))
    rf_out  = rf_model.transform(base_df).select("Date", col("prediction").alias("rf_pred"))
    gbt_out = gbt_model.transform(base_df).select("Date", col("prediction").alias("gbt_pred"))

    # Join predictions
    final = (
        lr_out
        .join(rf_out, on="Date")
        .join(gbt_out, on="Date")
        .withColumn("lr_correct",  col("lr_pred")  == col("Target"))
        .withColumn("rf_correct",  col("rf_pred")  == col("Target"))
        .withColumn("gbt_correct", col("gbt_pred") == col("Target"))
    )

    print(f"\n===== AAPL | Batch {batch_id} =====")
    final.show(truncate=False)

    base_df.unpersist()

query = (
    df.writeStream
    .foreachBatch(process_batch)
    .start()
)

query.awaitTermination()
