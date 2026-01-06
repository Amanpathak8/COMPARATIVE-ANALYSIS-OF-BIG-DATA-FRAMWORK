import pandas as pd
import numpy as np
import joblib, os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------
# DATASETS TO TRAIN ON
# --------------------------------
DATASETS = {
    "AAPL": "data_full_aapl.csv",
    "MSFT": "data_full_msft.csv"
}

os.makedirs("models/save_models", exist_ok=True)
os.makedirs("results", exist_ok=True)

for name, path in DATASETS.items():

    print(f"\n==============================")
    print(f" Training Logistic Regression → {name}")
    print(f"==============================")

    df = pd.read_csv(path)

    # Convert Date column back to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=["Date"])

    # Features and target
    X = df.drop(columns=["Target", "Date"])
    y = df["Target"]

    # Train-test split: last 10% = test
    split = int(len(df) * 0.9)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test = df['Date'][split:]

    # Train Logistic Regression
    model = LogisticRegression(max_iter=400)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, pred)

    print(f"\n📌 Logistic Regression Evaluation ({name})")
    print(f"Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, pred, zero_division=0))

    # Save last 10 predictions
    results = pd.DataFrame({
        "Date": dates_test.tail(10).dt.strftime("%d-%m-%Y"),
        "Actual": y_test.tail(10).values,
        "Predicted": pred[-10:]
    })

    results["Match"] = np.where(
        results["Actual"] == results["Predicted"],
        "✔ Correct",
        "❌ Incorrect"
    )

    results_file = f"results/logistic_results_{name}.csv"
    results.to_csv(results_file, index=False)

    print("\n📘 Sample Predictions (last 10 rows):")
    print(results.to_string(index=False))

    # Save model
    model_path = f"models/save_models/model_logistic_{name}.pkl"
    joblib.dump(model, model_path)

    print(f"\n✔ Saved Logistic model → {model_path}")
    print(f"✔ Saved results → {results_file}")
