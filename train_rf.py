# train_rf.py

import pandas as pd
import numpy as np
import joblib, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data_full.csv")

# Ensure Date column is datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])

X = df.drop(columns=["Target", "Date"])
y = df["Target"]

# Train-test split (last 10%)
split = int(len(df) * 0.9)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_test = df["Date"][split:]

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("\n📌 Random Forest Evaluation")
print(f"Accuracy: {acc:.4f}\n")
print(classification_report(y_test, pred, zero_division=0))

# Prepare output table
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

print("\n📘 Sample Predictions (last 10 rows):")
print(results.to_string(index=False))

# Save model
os.makedirs("models/save_models", exist_ok=True)
joblib.dump(model, "models/save_models/model_rf.pkl")
print("\n✔ Saved RF model → models/save_models/model_rf.pkl")
