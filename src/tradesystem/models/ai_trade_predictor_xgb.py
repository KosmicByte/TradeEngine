
import yfinance as yf
import pandas as pd
import numpy as np
import talib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# --- CONFIGURATION ---
INDEX = "^NSEI"
INTERVAL = "15m"
PERIOD = "60d"
MODEL_FILE = "trade_predictor_xgb.pkl"

# --- FEATURE ENGINEERING ---
def fetch_and_engineer_data():
    df = yf.download(INDEX, interval=INTERVAL, period=PERIOD)
    df.dropna(inplace=True)
    df["RSI"] = talib.RSI(df["Close"], timeperiod=14)
    df["MACD"], df["MACD_Signal"], _ = talib.MACD(df["Close"])
    df["EMA20"] = talib.EMA(df["Close"], timeperiod=20)
    df["EMA50"] = talib.EMA(df["Close"], timeperiod=50)
    df["Volume_MA"] = df["Volume"].rolling(window=10).mean()
    df["Volume_Boost"] = df["Volume"] / df["Volume_MA"]
    df.dropna(inplace=True)

    df["Label"] = 0
    for i in range(10, len(df)-5):
        rr_up = (df["Close"].iloc[i+5] - df["Close"].iloc[i]) / (df["Close"].iloc[i] - df["Low"].iloc[i-5:i].min())
        rr_down = (df["Close"].iloc[i] - df["Close"].iloc[i+5]) / (df["High"].iloc[i-5:i].max() - df["Close"].iloc[i])
        if rr_up >= 2:
            df.loc[df.index[i], "Label"] = 1  # BUY worked
        elif rr_down >= 2:
            df.loc[df.index[i], "Label"] = 1  # SELL worked
        else:
            df.loc[df.index[i], "Label"] = 0

    features = df[["RSI", "MACD", "MACD_Signal", "EMA20", "EMA50", "Volume_Boost"]]
    labels = df["Label"]
    return features, labels

# --- MODEL TRAINING ---
def train_model():
    X, y = fetch_and_engineer_data()
    X.fillna(0, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print("📊 Model Evaluation:")
    print(classification_report(y_test, y_pred))

    joblib.dump((scaler, model), MODEL_FILE)
    print(f"✅ Model saved to: {MODEL_FILE}")

# --- MODEL PREDICTION FOR LIVE USAGE ---
def predict_signals(live_df):
    if not os.path.exists(MODEL_FILE):
        print("❌ Model not found. Please train it first.")
        return []

    scaler, model = joblib.load(MODEL_FILE)

    features = live_df[["RSI", "MACD", "MACD_Signal", "EMA20", "EMA50", "Volume_Boost"]].fillna(0)
    features_scaled = scaler.transform(features)
    preds = model.predict(features_scaled)

    live_df["Prediction"] = preds
    return live_df[live_df["Prediction"] == 1]

if __name__ == "__main__":
    train_model()
