
import os
import pandas as pd
import numpy as np
import yfinance as yf
import talib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

MODEL_FILE = "trade_predictor_xgb.pkl"
PERFORMANCE_LOG = "model_performance_log.csv"

def fetch_data(symbol="^NSEI", period="60d", interval="15m"):
    df = yf.download(symbol, period=period, interval=interval)
    df["RSI"] = talib.RSI(df["Close"], 14)
    df["MACD"], df["MACD_Signal"], _ = talib.MACD(df["Close"])
    df["EMA20"] = talib.EMA(df["Close"], timeperiod=20)
    df["EMA50"] = talib.EMA(df["Close"], timeperiod=50)
    df["Volume_MA"] = df["Volume"].rolling(window=10).mean()
    df["Volume_Boost"] = df["Volume"] / df["Volume_MA"]
    df.dropna(inplace=True)

    df["Label"] = 0
    for i in range(10, len(df) - 5):
        rr_up = (df["Close"].iloc[i + 5] - df["Close"].iloc[i]) / (df["Close"].iloc[i] - df["Low"].iloc[i - 5:i].min())
        rr_down = (df["Close"].iloc[i] - df["Close"].iloc[i + 5]) / (df["High"].iloc[i - 5:i].max() - df["Close"].iloc[i])
        if rr_up >= 2 or rr_down >= 2:
            df.loc[df.index[i], "Label"] = 1
    return df

def train_and_log(df):
    X = df[["RSI", "MACD", "MACD_Signal", "EMA20", "EMA50", "Volume_Boost"]].fillna(0)
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save model and scaler
    joblib.dump((scaler, model), MODEL_FILE)

    # Log metrics
    metrics = {
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Accuracy": acc,
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1": report["1"]["f1-score"]
    }

    df_log = pd.DataFrame([metrics])
    if os.path.exists(PERFORMANCE_LOG):
        df_log.to_csv(PERFORMANCE_LOG, mode="a", header=False, index=False)
    else:
        df_log.to_csv(PERFORMANCE_LOG, index=False)

    print(f"✅ Weekly model updated. Accuracy: {acc:.2f}, Log saved to {PERFORMANCE_LOG}")

if __name__ == "__main__":
    df = fetch_data()
    train_and_log(df)
