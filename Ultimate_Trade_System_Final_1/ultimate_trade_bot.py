
import yfinance as yf
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# --- CONFIGURATION ---
INDEX = "^NSEI"  # Nifty. You can switch to "^NSEBANK" or "^CNXFINSERV"
INTERVAL = "15m"
PERIOD = "15d"
RR_THRESHOLD = 2.0
MIN_VOLUME_BOOST = 1.3

# --- FETCH DATA ---
def fetch_data():
    df = yf.download(INDEX, interval=INTERVAL, period=PERIOD)
    df.dropna(inplace=True)
    df["RSI"] = talib.RSI(df["Close"], timeperiod=14)
    df["MACD"], df["MACD_Signal"], _ = talib.MACD(df["Close"])
    df["EMA20"] = talib.EMA(df["Close"], timeperiod=20)
    df["EMA50"] = talib.EMA(df["Close"], timeperiod=50)
    df["Volume_MA"] = df["Volume"].rolling(window=10).mean()
    df["Volume_Boost"] = df["Volume"] / df["Volume_MA"]
    return df

# --- SIGNAL ENGINE ---
def detect_signals(df):
    signals = []
    for i in range(50, len(df)):
        row = df.iloc[i]
        if np.isnan([row["RSI"], row["MACD"], row["MACD_Signal"], row["Volume_MA"]]).any():
            continue

        close = row["Close"]
        time = row.name
        if row["RSI"] < 30 and row["MACD"] > row["MACD_Signal"] and row["Volume_Boost"] > MIN_VOLUME_BOOST:
            sl = df["Low"].iloc[i-5:i].min()
            target = close + 2 * (close - sl)
            rr = (target - close) / (close - sl)
            if rr >= RR_THRESHOLD:
                signals.append(["BUY", time, close, sl, target, rr])
        elif row["RSI"] > 70 and row["MACD"] < row["MACD_Signal"] and row["Volume_Boost"] > MIN_VOLUME_BOOST:
            sl = df["High"].iloc[i-5:i].max()
            target = close - 2 * (sl - close)
            rr = (close - target) / (sl - close)
            if rr >= RR_THRESHOLD:
                signals.append(["SELL", time, close, sl, target, rr])
    return signals

# --- OPTION LEG SUGGESTION ---
def suggest_option_leg(signal_type, spot):
    strike = round(spot / 100) * 100
    if signal_type == "BUY":
        return f"Buy CE @ Strike {strike + 100}"
    elif signal_type == "SELL":
        return f"Buy PE @ Strike {strike - 100}"
    return "No suggestion"

# --- LOG & CHART GENERATION ---
def save_log_and_plot(df, signals):
    df_trades = pd.DataFrame(signals, columns=["Signal", "Time", "Entry", "SL", "Target", "RR"])
    df_trades["Option Suggestion"] = df_trades.apply(lambda x: suggest_option_leg(x["Signal"], x["Entry"]), axis=1)

    log_file = f"trade_log_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    df_trades.to_excel(log_file, index=False)

    df["Signal"] = None
    for s in signals:
        df.loc[s[1], "Signal"] = s[0]

    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(df["Close"], label="Close", color="blue")
    ax.plot(df["EMA20"], label="EMA20", linestyle="--", color="orange")
    ax.plot(df["EMA50"], label="EMA50", linestyle="--", color="green")
    buy_signals = df[df["Signal"] == "BUY"]
    sell_signals = df[df["Signal"] == "SELL"]
    ax.scatter(buy_signals.index, buy_signals["Close"], color="lime", marker="^", label="BUY")
    ax.scatter(sell_signals.index, sell_signals["Close"], color="red", marker="v", label="SELL")
    ax.legend()
    ax.grid()
    plt.title("Trade Signals with EMA and Volume Boost")
    chart_file = f"trade_chart_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    plt.savefig(chart_file)
    plt.close()
    return log_file, chart_file

# --- MAIN ---
def main():
    df = fetch_data()
    signals = detect_signals(df)
    if signals:
        log_file, chart_file = save_log_and_plot(df, signals)
        print(f"✅ {len(signals)} Trades Logged. Log saved to {log_file}")
    else:
        print("No valid trade setups found.")

if __name__ == "__main__":
    main()
