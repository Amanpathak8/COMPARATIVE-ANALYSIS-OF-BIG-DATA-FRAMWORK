import os
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "data_full_msft.csv")
MODEL_DIR = os.path.join(BASE_DIR, "flink_models")

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Date", "Target"])
y = df["Target"]

# =====================================================
# TRAIN / TEST SPLIT (90 / 10)
# =====================================================
split = int(len(df) * 0.9)

X_train = X.iloc[:split]
X_test  = X.iloc[split:]

y_train = y.iloc[:split]
y_test  = y.iloc[split:]

# =====================================================
# LOGISTIC REGRESSION
# =====================================================
lr = LogisticRegression(max_iter=300)
lr.fit(X_train, y_train)
joblib.dump(lr, os.path.join(MODEL_DIR, "lr_MSFT.pkl"))

# =====================================================
# RANDOM FOREST
# =====================================================
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)
rf.fit(X_train, y_train)
joblib.dump(rf, os.path.join(MODEL_DIR, "rf_MSFT.pkl"))

# =====================================================
# XGBOOST
# =====================================================
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train, y_train)
joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_MSFT.pkl"))

# =====================================================
# LSTM (SEQUENCE MODEL)
# =====================================================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, os.path.join(MODEL_DIR, "lstm_scaler_MSFT.pkl"))

def make_sequences(X, y, seq_len=5):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y.iloc[i+seq_len])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = make_sequences(X_scaled, y)

split_seq = int(len(X_seq) * 0.9)

X_train_seq = X_seq[:split_seq]
X_test_seq  = X_seq[split_seq:]

y_train_seq = y_seq[:split_seq]
y_test_seq  = y_seq[split_seq:]

lstm = Sequential([
    LSTM(32, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dense(1, activation="sigmoid")
])

lstm.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

lstm.fit(
    X_train_seq,
    y_train_seq,
    epochs=20,
    batch_size=16,
    verbose=1,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
)

lstm.save(os.path.join(MODEL_DIR, "lstm_MSFT.h5"))

# =====================================================
# DONE
# =====================================================
print("\n✔ MSFT models trained successfully")
print("Saved models:")
print(" - lr_MSFT.pkl")
print(" - rf_MSFT.pkl")
print(" - xgb_MSFT.pkl")
print(" - lstm_MSFT.h5")
print(" - lstm_scaler_MSFT.pkl")
