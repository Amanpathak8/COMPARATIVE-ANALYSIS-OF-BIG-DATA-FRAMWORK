import pandas as pd
import json
import time
from kafka import KafkaProducer

BOOTSTRAP_SERVER = "localhost:9092"

# -------------------------------
# DATASETS & TOPICS
# -------------------------------
DATASETS = {
    "AAPL": {
        "path": "data_full_aapl.csv",
        "topic": "aapl_stream"
    },
    "MSFT": {
        "path": "data_full_msft.csv",
        "topic": "msft_stream"
    }
}

def connect_producer():
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

def produce_dataset(csv_path, topic, sleep_time=0.5):
    df = pd.read_csv(csv_path)

    producer = connect_producer()
    print(f"[OK] Producer connected → {BOOTSTRAP_SERVER}")
    print(f"[INFO] Streaming {csv_path} → topic: {topic}")

    for _, row in df.iterrows():
        data = row.to_dict()
        producer.send(topic, value=data)
        print(f"[SENT → {topic}] {data}")
        time.sleep(sleep_time)

    producer.flush()
    print(f"\n✔ Completed streaming for topic: {topic}\n")

if __name__ == "__main__":
    for name, cfg in DATASETS.items():
        print(f"\n==============================")
        print(f" Starting producer for {name}")
        print(f"==============================")
        produce_dataset(
            csv_path=cfg["path"],
            topic=cfg["topic"],
            sleep_time=0.5
        )
