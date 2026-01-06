from kafka import KafkaConsumer
import json

TOPIC = "aapl_stream"
BOOTSTRAP_SERVER = "localhost:9092"

def consume_data():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVER,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode('utf-8'))
    )

    print(f"[OK] Consumer connected → {BOOTSTRAP_SERVER}")
    print(f"Listening to topic: {TOPIC}\n")

    for message in consumer:
        data = message.value
        print(f"[RECEIVED] {data}")

if __name__ == "__main__":
    consume_data()
