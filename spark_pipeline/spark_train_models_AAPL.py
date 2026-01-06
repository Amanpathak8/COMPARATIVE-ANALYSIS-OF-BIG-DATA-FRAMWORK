from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline
import os

spark = SparkSession.builder.appName("Spark_Train_AAPL").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Load AAPL dataset
df = spark.read.csv("data_full_aapl.csv", header=True, inferSchema=True)

features = ["Open","High","Low","Close","Volume","SMA5","SMA10","Ret1","Ret5"]

assembler = VectorAssembler(
    inputCols=features,
    outputCol="features"
)

lr = LogisticRegression(labelCol="Target")
rf = RandomForestClassifier(labelCol="Target", numTrees=100)
gbt = GBTClassifier(labelCol="Target")

pipelines = {
    "lr_AAPL": Pipeline(stages=[assembler, lr]),
    "rf_AAPL": Pipeline(stages=[assembler, rf]),
    "gbt_AAPL": Pipeline(stages=[assembler, gbt])
}

os.makedirs("spark_models", exist_ok=True)

for name, pipe in pipelines.items():
    model = pipe.fit(df)
    model.write().overwrite().save(f"spark_models/{name}")
    print(f"Saved {name}")

spark.stop()
