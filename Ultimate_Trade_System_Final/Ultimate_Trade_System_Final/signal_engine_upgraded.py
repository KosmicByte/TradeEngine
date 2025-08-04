
import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    df["H-L"] = df["high"] - df["low"]
    df["H-PC"] = abs(df["high"] - df["close"].shift(1))
    df["L-PC"] = abs(df["low"] - df["close"].shift(1))
    tr = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def detect_structure(df):
    df["BOS"] = (df["high"] > df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    df["CHoCH"] = (df["high"] < df["high"].shift(1)) & (df["low"] < df["low"].shift(1))
    return df

def validate_multi_timeframe(df_15m, df_1h):
    # Check trend agreement: if both 15m and 1h are in uptrend (BOS) or downtrend (CHoCH)
    last_15 = df_15m.iloc[-1]
    last_1h = df_1h.iloc[-1]
    if last_15["BOS"] and last_1h["BOS"]:
        return "Bullish"
    elif last_15["CHoCH"] and last_1h["CHoCH"]:
        return "Bearish"
    else:
        return "Neutral"

def find_support_resistance(df, window=20):
    highs = df["high"].rolling(window).max()
    lows = df["low"].rolling(window).min()
    return highs, lows

def generate_signals(df_15m, df_1h):
    df_15m["ATR"] = calculate_atr(df_15m)
    df_15m = detect_structure(df_15m)
    df_1h = detect_structure(df_1h)
    bias = validate_multi_timeframe(df_15m, df_1h)
    sr_high, sr_low = find_support_resistance(df_15m)

    signals = []
    for i in range(20, len(df_15m)):
        row = df_15m.iloc[i]
        if bias == "Bullish" and row["BOS"] and row["close"] > sr_high[i-1]:
            rr = (sr_low[i-1] - row["close"]) / (row["close"] - sr_high[i-1]) if row["close"] - sr_high[i-1] != 0 else 0
            if rr >= 2:
                signals.append(("Buy", i, row["close"]))
        elif bias == "Bearish" and row["CHoCH"] and row["close"] < sr_low[i-1]:
            rr = (sr_high[i-1] - row["close"]) / (row["close"] - sr_low[i-1]) if row["close"] - sr_low[i-1] != 0 else 0
            if rr >= 2:
                signals.append(("Sell", i, row["close"]))
    return signals

# Example usage:
# df_15m = pd.read_csv("nifty_15m.csv")
# df_1h = pd.read_csv("nifty_1h.csv")
# signals = generate_signals(df_15m, df_1h)
# for signal in signals:
#     print("Signal:", signal)
