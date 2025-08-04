
import requests
import pandas as pd

def fetch_nifty_vix():
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=VIX"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }

    with requests.Session() as session:
        session.headers.update(headers)
        _ = session.get("https://www.nseindia.com")
        r = session.get(url)

    vix = r.json()["records"]["underlyingValue"]
    return float(vix)

def atr_trend_filter(df, window=14):
    df = df.copy()
    df["H-L"] = df["high"] - df["low"]
    df["H-PC"] = abs(df["high"] - df["close"].shift(1))
    df["L-PC"] = abs(df["low"] - df["close"].shift(1))
    tr = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["atr"] = tr.rolling(window=window).mean()

    df["atr_slope"] = df["atr"].diff()
    slope = df["atr_slope"].iloc[-5:].mean()

    if slope > 0:
        regime = "Volatile"
    else:
        regime = "Calm"

    return regime, df[["atr", "atr_slope"]].tail(1)
