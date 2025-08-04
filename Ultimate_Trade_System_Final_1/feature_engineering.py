
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

def add_features(df):
    df = df.copy()
    df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()
    macd = MACD(close=df["close"])
    df["macd"] = macd.macd_diff()
    df["ema_21"] = EMAIndicator(close=df["close"], window=21).ema_indicator()
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["atr"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"]).average_true_range()
    df["bb_percent"] = BollingerBands(close=df["close"]).bollinger_pband()
    df["volume_delta"] = df["volume"].pct_change().fillna(0)

    df["price_return"] = df["close"].pct_change().fillna(0)
    df["price_diff"] = df["close"].diff().fillna(0)

    # BOS/CHoCH flags from structure
    df["bos_flag"] = (df["high"] > df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    df["choch_flag"] = (df["high"] < df["high"].shift(1)) & (df["low"] < df["low"].shift(1))

    df = df.dropna()
    return df

# Example usage:
# df = pd.read_csv("nifty_15m.csv")
# enriched_df = add_features(df)
