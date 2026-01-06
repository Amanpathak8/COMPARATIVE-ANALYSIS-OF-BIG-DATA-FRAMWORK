from kafka import KafkaConsumer
import json
KAFKA_SERVER = "localhost:9092"
TOPIC = "aapl_stream"

def start_consumer():
    consumer = KafkaConsumer(TOPIC, bootstrap_servers=KAFKA_SERVER,
                             value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                             auto_offset_reset='earliest', enable_auto_commit=True)
    print("Listening...")
    for msg in consumer:
        print("Received:", msg.value)

if __name__ == '__main__':
    start_consumer()
