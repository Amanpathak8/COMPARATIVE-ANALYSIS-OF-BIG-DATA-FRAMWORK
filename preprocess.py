import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib, os

def preprocess(input_file, out_csv):
    # ---------------------------------
    # LOAD EXCEL OR CSV SAFELY
    # ---------------------------------
    if input_file.endswith(".xlsx"):
        df = pd.read_excel(input_file)
    elif input_file.endswith(".csv"):
        df = pd.read_csv(input_file)
    else:
        raise ValueError("Unsupported file format")

    # Fill missing values
    df = df.ffill().bfill()

    # ---------------------------------
    # SAFE DATE NORMALIZATION
    # ---------------------------------
    if 'Date' in df.columns:
        df['Date'] = df['Date'].astype(str)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])

    # ---------------------------------
    # FEATURE ENGINEERING
    # ---------------------------------
    df['SMA5'] = df['Close'].rolling(5, min_periods=1).mean()
    df['SMA10'] = df['Close'].rolling(10, min_periods=1).mean()
    df['Ret1'] = df['Close'].pct_change().fillna(0)
    df['Ret5'] = df['Close'].pct_change(5).fillna(0)

    # ---------------------------------
    # CLASSIFICATION TARGET
    # ---------------------------------
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna().reset_index(drop=True)

    # Feature columns
    feats = ['Open','High','Low','Close','Volume','SMA5','SMA10','Ret1','Ret5']

    os.makedirs("models/save_models", exist_ok=True)

    scaler = MinMaxScaler()
    df[feats] = scaler.fit_transform(df[feats])

    # Save scaler (dataset-specific)
    scaler_name = os.path.basename(out_csv).replace(".csv", "_scaler.pkl")
    joblib.dump(scaler, f"models/save_models/{scaler_name}")

    # Save processed data
    df.to_csv(out_csv, index=False)

    print(f"✔ Preprocessed dataset → {out_csv}")
    print(f"✔ Scaler saved → {scaler_name}")

if __name__ == "__main__":
    # Dataset 1 – AAPL
    preprocess(
        input_file="AAPL_stock.xlsx",
        out_csv="data_full_aapl.csv"
    )

    # Dataset 2 – MSFT
    preprocess(
        input_file="Microsoft_Stocks.csv",
        out_csv="data_full_msft.csv"
    )
