import os
import pandas as pd
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "data_full.csv")
MODEL_DIR = os.path.join(BASE_DIR, "flink_models", "AAPL")

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Target", "Date"])
y = df["Target"]

# Train-test split (same logic as Spark)
split = int(len(df) * 0.9)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =====================================================
# LOGISTIC REGRESSION
# =====================================================
lr = LogisticRegression(max_iter=300)
lr.fit(X_train, y_train)

joblib.dump(lr, os.path.join(MODEL_DIR, "lr_AAPL.pkl"))

# =====================================================
# RANDOM FOREST
# =====================================================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
rf.fit(X_train, y_train)

joblib.dump(rf, os.path.join(MODEL_DIR, "rf_AAPL.pkl"))

# =====================================================
# XGBOOST (GRADIENT BOOSTING)
# =====================================================
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train, y_train)

joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_AAPL.pkl"))

# =====================================================
# LSTM MODEL
# =====================================================
# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, os.path.join(MODEL_DIR, "lstm_scaler_AAPL.pkl"))

# Create sequences (time-steps)
def create_sequences(X, y, window=10):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i])
        ys.append(y.iloc[i])
    return np.array(Xs), np.array(ys)

WINDOW = 10
X_seq, y_seq = create_sequences(X_scaled, y, WINDOW)

split_seq = int(len(X_seq) * 0.9)
X_train_seq, y_train_seq = X_seq[:split_seq], y_seq[:split_seq]

# Build LSTM
lstm = Sequential([
    LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dense(1, activation="sigmoid")
])

lstm.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

lstm.fit(
    X_train_seq,
    y_train_seq,
    epochs=15,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=3)],
    verbose=1
)

lstm.save(os.path.join(MODEL_DIR, "lstm_AAPL"))

# =====================================================
# DONE
# =====================================================
print("\n✔ Flink AAPL models trained and saved")
print("  - lr_AAPL.pkl")
print("  - rf_AAPL.pkl")
print("  - xgb_AAPL.pkl")
print("  - lstm_AAPL/")
print("  - lstm_scaler_AAPL.pkl")
