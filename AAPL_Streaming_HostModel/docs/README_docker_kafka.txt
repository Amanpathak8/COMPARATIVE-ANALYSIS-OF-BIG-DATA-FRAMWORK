This package contains Kafka + Spark + Flink configuration for running the streaming layer.
Model server expected to run on the host at http://localhost:5000 (inside containers use host.docker.internal).

How to use:
1) Place your preprocessed CSV 'data_full.csv' in the package root (next to docker/ folder).
2) From the docker folder run: docker-compose up -d
3) Create Kafka topic (inside kafka container):
   docker exec -it kafka bash
   kafka-topics --bootstrap-server kafka:9092 --create --topic aapl_stream --partitions 1 --replication-factor 1
4) From host, start the model server:
   python model_server.py  (ensure models are saved in models/save_models/)
5) Start consumer (optional host consumer for debugging):
   python kafka_layer/kafka_consumer.py
6) Start producer (from host):
   python kafka_layer/kafka_producer.py
7) To run Spark streaming inside container:
   docker exec -it spark bash
   # inside container: python3 spark_pipeline/spark_stream.py
   # or use spark-submit if spark present: /opt/spark/bin/spark-submit --master local[*] spark_pipeline/spark_stream.py
8) To run Flink inside container:
   docker exec -it flink bash
   # inside container: python3 flink_pipeline/flink_stream.py

Notes:
- The consumer inside Spark and Flink will call the model server at host.docker.internal:5000.
- If Docker Desktop doesn't support host.docker.internal, run model_server inside Docker (change MODEL_SERVER env to http://model_server:5000 and add model_server service to docker-compose).
