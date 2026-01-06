import pandas as pd
import json, time
from kafka import KafkaProducer

KAFKA_SERVER = "localhost:9092"  # running on host via docker-compose mapping
TOPIC = "aapl_stream"

def serializer(data):
    return json.dumps(data).encode("utf-8")

def start_producer(rate_per_sec=1):
    print(f"Connecting to Kafka at {KAFKA_SERVER} ...")
    producer = KafkaProducer(bootstrap_servers=KAFKA_SERVER, value_serializer=serializer)
    df = pd.read_csv("data_full.csv")  # ensure preprocess saved this
    for _, row in df.iterrows():
        msg = {
            "Date": str(row.get("Date")),
            "Open": float(row["Open"]),
            "High": float(row["High"]),
            "Low": float(row["Low"]),
            "Close": float(row["Close"]),
            "Volume": float(row["Volume"]),
            "SMA5": float(row.get("SMA5", 0)),
            "SMA10": float(row.get("SMA10", 0)),
            "Ret1": float(row.get("Ret1", 0)),
            "Ret5": float(row.get("Ret5", 0))
        }
        producer.send(TOPIC, msg)
        print("Sent:", msg)
        time.sleep(1.0/max(rate_per_sec,1))
    producer.flush()
    print("Finished sending.")

if __name__ == '__main__':
    start_producer(rate_per_sec=1)
