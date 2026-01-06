import pandas as pd
import numpy as np
import os, joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# DATASETS
# -------------------------------
DATASETS = {
    "AAPL": "data_full_aapl.csv",
    "MSFT": "data_full_msft.csv"
}

SEQ_LEN = 5

os.makedirs("models/save_models", exist_ok=True)
os.makedirs("results", exist_ok=True)

def make_sequences(X, y, seq=SEQ_LEN):
    Xs, ys = [], []
    for i in range(len(X) - seq):
        Xs.append(X[i:i + seq])
        ys.append(y[i + seq])
    return np.array(Xs), np.array(ys)

# -------------------------------
# TRAIN LSTM FOR EACH DATASET
# -------------------------------
for name, path in DATASETS.items():

    print("\n===================================")
    print(f" Training LSTM model for {name}")
    print("===================================")

    df = pd.read_csv(path)

    # Date handling
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Features & target
    X_raw = df.drop(columns=["Target", "Date"]).values
    y_raw = df["Target"].values

    # Scale features (dataset-specific)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    scaler_path = f"models/save_models/lstm_scaler_{name}.pkl"
    joblib.dump(scaler, scaler_path)

    # Create sequences
    X_seq, y_seq = make_sequences(X_scaled, y_raw)

    # Train-test split
    split = int(len(X_seq) * 0.9)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # Align dates
    dates_test = df["Date"].iloc[split + SEQ_LEN : split + SEQ_LEN + len(X_test)]

    # LSTM model
    model = Sequential([
        LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=16,
        verbose=1
    )

    # Predictions
    pred = (model.predict(X_test) > 0.5).astype(int).reshape(-1)

    acc = accuracy_score(y_test, pred)

    print(f"\n📌 LSTM Evaluation ({name})")
    print(f"Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, pred, zero_division=0))

    # Last 10 predictions
    results = pd.DataFrame({
        "Date": dates_test.tail(10).dt.strftime("%d-%m-%Y"),
        "Actual": y_test[-10:],
        "Predicted": pred[-10:]
    })

    results["Match"] = np.where(
        results["Actual"] == results["Predicted"],
        "✔ Correct",
        "❌ Incorrect"
    )

    results_file = f"results/lstm_results_{name}.csv"
    results.to_csv(results_file, index=False)

    print("\n📘 Sample Predictions (last 10 rows):")
    print(results.to_string(index=False))

    # Save model
    model_path = f"models/save_models/model_lstm_{name}.h5"
    model.save(model_path)

    print(f"\n✔ Saved LSTM model → {model_path}")
    print(f"✔ Saved LSTM scaler → {scaler_path}")
    print(f"✔ Saved results → {results_file}")
