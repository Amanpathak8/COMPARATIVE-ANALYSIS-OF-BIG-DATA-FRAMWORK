import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.pipeline import PipelineModel

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "spark_models")

# =====================================================
# SPARK SESSION
# =====================================================
spark = (
    SparkSession.builder
    .appName("Spark_Streaming_Predict_MSFT")
    .config("spark.sql.shuffle.partitions", "1")
    .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# =====================================================
# LOAD MODELS
# =====================================================
print("[INFO] Loading MSFT models...")

lr_model  = PipelineModel.load(os.path.join(MODEL_DIR, "lr_MSFT"))
rf_model  = PipelineModel.load(os.path.join(MODEL_DIR, "rf_MSFT"))
gbt_model = PipelineModel.load(os.path.join(MODEL_DIR, "gbt_MSFT"))

print("[OK] MSFT models loaded")

# =====================================================
# SCHEMA
# =====================================================
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

# =====================================================
# READ KAFKA STREAM
# =====================================================
raw_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "msft_stream")
    .option("startingOffsets", "latest")
    .load()
)

parsed_df = (
    raw_df
    .selectExpr("CAST(value AS STRING)")
    .select(from_json(col("value"), schema).alias("data"))
    .select("data.*")
    .withColumn("ingest_time", current_timestamp())
)

# =====================================================
# SAFE BATCH PROCESSING (NO count())
# =====================================================
def process_batch(batch_df, batch_id):
    if batch_df.rdd.isEmpty():
        return

    out = lr_model.transform(batch_df) \
        .withColumnRenamed("prediction", "lr_pred") \
        .drop("features", "rawPrediction", "probability")

    out = rf_model.transform(out) \
        .withColumnRenamed("prediction", "rf_pred") \
        .drop("features", "rawPrediction", "probability")

    out = gbt_model.transform(out) \
        .withColumnRenamed("prediction", "gbt_pred") \
        .drop("features", "rawPrediction", "probability")

    final_df = (
        out
        .withColumn("processing_time", current_timestamp())
        .withColumn(
            "latency_ms",
            (unix_timestamp("processing_time") -
             unix_timestamp("ingest_time")) * 1000
        )
    )

    final_df.select(
        "Date",
        "lr_pred",
        "rf_pred",
        "gbt_pred",
        "Target",
        "latency_ms"
    ).show(truncate=False)

# =====================================================
# START STREAM (REAL-TIME STYLE)
# =====================================================
query = (
    parsed_df
    .writeStream
    .foreachBatch(process_batch)
    .trigger(processingTime="1 second")
    .option("checkpointLocation", "checkpoint_msft")
    .start()
)

query.awaitTermination()
