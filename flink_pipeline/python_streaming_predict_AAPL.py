import json
import time
import numpy as np
from kafka import KafkaConsumer
import joblib
from tensorflow.keras.models import load_model

# =====================================================
# LOAD MODELS
# =====================================================
print("[INFO] Loading AAPL ML models...")

lr_model = joblib.load("models/save_models/model_logistic.pkl")
rf_model = joblib.load("models/save_models/model_rf.pkl")
xgb_model = joblib.load("models/save_models/model_xgb.pkl")
lstm_model = load_model("models/save_models/model_lstm.h5")
scaler = joblib.load("models/save_models/scaler.pkl")

print("[OK] Models loaded successfully")

# =====================================================
# KAFKA CONSUMER
# =====================================================
consumer = KafkaConsumer(
    "aapl_stream",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="latest",
    value_deserializer=lambda v: json.loads(v.decode("utf-8"))
)

print("[INFO] Listening to Kafka topic: aapl_stream")

# =====================================================
# STREAMING LOOP
# =====================================================
for msg in consumer:
    record = msg.value

    try:
        # Feature order MUST match training
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

        # Scale features
        features_scaled = scaler.transform(features)

        # Predictions
        lr_pred = int(lr_model.predict(features_scaled)[0])
        rf_pred = int(rf_model.predict(features_scaled)[0])
        xgb_pred = int(xgb_model.predict(features_scaled)[0])
        lstm_pred = int(
            (lstm_model.predict(features_scaled.reshape(1, 1, -1)) > 0.5)[0][0]
        )

        print("\n📊 REAL-TIME AAPL PREDICTION")
        print(f"Date       : {record['Date']}")
        print(f"LR Prediction   : {lr_pred}")
        print(f"RF Prediction   : {rf_pred}")
        print(f"XGB Prediction  : {xgb_pred}")
        print(f"LSTM Prediction : {lstm_pred}")

    except Exception as e:
        print("[ERROR]", e)

    time.sleep(0.2)
