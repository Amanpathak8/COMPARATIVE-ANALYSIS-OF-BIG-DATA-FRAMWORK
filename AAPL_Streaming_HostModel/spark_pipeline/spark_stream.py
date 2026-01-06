from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import requests, json, os, time

MODEL_SERVER = os.environ.get('MODEL_SERVER', 'http://host.docker.internal:5000')  # host model server
KAFKA_BOOT = os.environ.get('KAFKA_BOOT', 'kafka:9092')  # inside container use kafka:9092

schema = StructType([
    StructField('Date', StringType()),
    StructField('Open', DoubleType()),
    StructField('High', DoubleType()),
    StructField('Low', DoubleType()),
    StructField('Close', DoubleType()),
    StructField('Volume', DoubleType()),
    StructField('SMA5', DoubleType()),
    StructField('SMA10', DoubleType()),
    StructField('Ret1', DoubleType()),
    StructField('Ret5', DoubleType())
])

def call_model(feats):
    payload = {'model':'logistic','features':feats}
    try:
        r = requests.post(MODEL_SERVER + '/predict', json=payload, timeout=5)
        return int(r.json().get('prediction', -1))
    except Exception as e:
        return -1

if __name__ == '__main__':
    spark = SparkSession.builder.appName('spark_stream').getOrCreate()
    df = spark.readStream.format('kafka') \
        .option('kafka.bootstrap.servers', KAFKA_BOOT) \
        .option('subscribe', 'aapl_stream') \
        .option('startingOffsets', 'earliest') \
        .load()
    json_df = df.selectExpr("CAST(value AS STRING) as json_str")
    parsed = json_df.select(from_json(col('json_str'), schema).alias('data')).select('data.*')

    def predict_udf(open,high,low,close,volume,sma5,sma10,ret1,ret5):
        feats = [open,high,low,close,volume,sma5 or 0.0,sma10 or 0.0,ret1 or 0.0,ret5 or 0.0]
        return call_model(feats)

    p_udf = udf(predict_udf, IntegerType())
    out = parsed.withColumn('prediction', p_udf(col('Open'),col('High'),col('Low'),col('Close'),col('Volume'),col('SMA5'),col('SMA10'),col('Ret1'),col('Ret5')))
    query = out.writeStream.outputMode('append').format('console').start()
    query.awaitTermination()
