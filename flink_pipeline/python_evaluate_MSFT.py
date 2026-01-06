import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

# ===============================
# LOAD MSFT DATA
# ===============================
DATA_PATH = "data/data_full_msft.csv"
df = pd.read_csv(DATA_PATH)

split = int(len(df) * 0.9)
test_df = df.iloc[split:]

X_test = test_df.drop(columns=["Target", "Date"])
y_test = test_df["Target"].values

# ===============================
# LOAD MODELS
# ===============================
lr_model = joblib.load("models/save_models/model_logistic.pkl")
rf_model = joblib.load("models/save_models/model_rf.pkl")
xgb_model = joblib.load("models/save_models/model_xgb.pkl")
lstm_model = load_model("models/save_models/model_lstm.h5")
scaler = joblib.load("models/save_models/scaler.pkl")

X_scaled = scaler.transform(X_test)

# ===============================
# PREDICTIONS
# ===============================
lr_pred = lr_model.predict(X_scaled)
rf_pred = rf_model.predict(X_scaled)
xgb_pred = xgb_model.predict(X_scaled)
lstm_pred = (
    lstm_model.predict(X_scaled.reshape(-1, 1, X_scaled.shape[1])) > 0.5
).astype(int).flatten()

# ===============================
# METRICS FUNCTION
# ===============================
def metrics(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0)
    )

results = {
    "Logistic Regression": metrics(y_test, lr_pred),
    "Random Forest": metrics(y_test, rf_pred),
    "XGBoost": metrics(y_test, xgb_pred),
    "LSTM": metrics(y_test, lstm_pred)
}

# ===============================
# DISPLAY TABLE
# ===============================
table = pd.DataFrame(
    results,
    index=["Accuracy", "Precision", "Recall", "F1-Score"]
).T.round(4)

print("\nPrediction Performance on MSFT Dataset\n")
print(table)
