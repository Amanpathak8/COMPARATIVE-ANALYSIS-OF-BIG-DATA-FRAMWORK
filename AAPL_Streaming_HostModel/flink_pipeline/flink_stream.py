from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
import json, requests, os, time

MODEL_SERVER = os.environ.get('MODEL_SERVER', 'http://host.docker.internal:5000')
KAFKA_BOOT = os.environ.get('KAFKA_BOOT', 'kafka:9092')
TOPIC = 'aapl_stream'

def process(record):
    try:
        obj = json.loads(record)
        feats = [obj.get('Open'), obj.get('High'), obj.get('Low'), obj.get('Close'), obj.get('Volume'), obj.get('SMA5',0), obj.get('SMA10',0), obj.get('Ret1',0), obj.get('Ret5',0)]
        payload = {'model':'logistic','features':feats}
        r = requests.post(MODEL_SERVER + '/predict', json=payload, timeout=5)
        print('Flink pred', r.json(), 'for', obj)
    except Exception as e:
        print('error', e)

def main():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    props = {'bootstrap.servers': KAFKA_BOOT, 'group.id': 'flink-group'}
    consumer = FlinkKafkaConsumer(TOPIC, SimpleStringSchema(), props)
    ds = env.add_source(consumer)
    ds.map(lambda x: process(x))
    env.execute('flink_stream_job')

if __name__ == '__main__':
    main()
