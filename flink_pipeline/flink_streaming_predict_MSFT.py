import json
import joblib
import numpy as np
from tensorflow.keras.models import load_model

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types

# =====================================================
# LOAD TRAINED MODELS (ONCE)
# =====================================================
MODEL_DIR = "flink_models"

lr_model = joblib.load(f"{MODEL_DIR}/lr_MSFT.pkl")
rf_model = joblib.load(f"{MODEL_DIR}/rf_MSFT.pkl")
xgb_model = joblib.load(f"{MODEL_DIR}/xgb_MSFT.pkl")
lstm_model = load_model(f"{MODEL_DIR}/lstm_MSFT.h5")
scaler = joblib.load(f"{MODEL_DIR}/lstm_scaler_MSFT.pkl")

SEQ_LEN = 5
sequence_buffer = []

# =====================================================
# PREDICTION FUNCTION
# =====================================================
def predict_event(value):
    global sequence_buffer

    record = json.loads(value)

    features = np.array([[
        record["Open"],
        record["High"],
        record["Low"],
        record["Close"],
        record["Volume"],
        record["SMA5"],
        record["SMA10"],
        record["Ret1"],
        record["Ret5"]
    ]])

    # ML predictions
    lr_pred = int(lr_model.predict(features)[0])
    rf_pred = int(rf_model.predict(features)[0])
    xgb_pred = int(xgb_model.predict(features)[0])

    # LSTM prediction
    scaled = scaler.transform(features)
    sequence_buffer.append(scaled[0])

    lstm_pred = None
    if len(sequence_buffer) >= SEQ_LEN:
        seq_input = np.array(sequence_buffer[-SEQ_LEN:]).reshape(1, SEQ_LEN, -1)
        lstm_pred = int((lstm_model.predict(seq_input, verbose=0)[0][0]) > 0.5)

    return json.dumps({
        "Date": record["Date"],
        "LR_Prediction": lr_pred,
        "RF_Prediction": rf_pred,
        "XGB_Prediction": xgb_pred,
        "LSTM_Prediction": lstm_pred
    })

# =====================================================
# FLINK ENVIRONMENT
# =====================================================
env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(1)

# =====================================================
# KAFKA SOURCE
# =====================================================
kafka_source = KafkaSource.builder() \
    .set_bootstrap_servers("localhost:9092") \
    .set_topics("msft_stream") \
    .set_group_id("flink-msft-group") \
    .set_value_only_deserializer(SimpleStringSchema()) \
    .build()

stream = env.from_source(
    kafka_source,
    watermark_strategy=None,
    source_name="Kafka MSFT Source"
)

# =====================================================
# APPLY PREDICTIONS
# =====================================================
predictions = stream.map(
    predict_event,
    output_type=Types.STRING()
)

# =====================================================
# OUTPUT (CONSOLE)
# =====================================================
predictions.print()

# =====================================================
# EXECUTE
# =====================================================
env.execute("Flink Real-Time MSFT Prediction")
