from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("Spark_Batch_Training").getOrCreate()

FEATURES = ["Open","High","Low","Close","Volume","SMA5","SMA10","Ret1","Ret5"]

DATASETS = {
    "AAPL": "data_full_aapl.csv",
    "MSFT": "data_full_msft.csv"
}

for name, path in DATASETS.items():
    print(f"\nTraining Spark ML models for {name}")

    df = spark.read.csv(path, header=True, inferSchema=True).dropna()

    assembler = VectorAssembler(inputCols=FEATURES, outputCol="features_raw")
    scaler = MinMaxScaler(inputCol="features_raw", outputCol="features")

    lr = LogisticRegression(featuresCol="features", labelCol="Target")
    rf = RandomForestClassifier(featuresCol="features", labelCol="Target", numTrees=200)
    gbt = GBTClassifier(featuresCol="features", labelCol="Target", maxIter=50)

    lr_pipe = Pipeline(stages=[assembler, scaler, lr])
    rf_pipe = Pipeline(stages=[assembler, scaler, rf])
    gbt_pipe = Pipeline(stages=[assembler, scaler, gbt])

    lr_model = lr_pipe.fit(df)
    rf_model = rf_pipe.fit(df)
    gbt_model = gbt_pipe.fit(df)

    lr_model.write().overwrite().save(f"spark_models/lr_{name}")
    rf_model.write().overwrite().save(f"spark_models/rf_{name}")
    gbt_model.write().overwrite().save(f"spark_models/gbt_{name}")

    print(f"✔ Saved Spark models for {name}")

spark.stop()
