
import pandas as pd
from datetime import timedelta

def label_past_trades(df, entry_price_col="entry_price", exit_price_col="exit_price", target=0.02, stoploss=0.01):
    """
    Label trade outcomes as: 1 = TP Hit, 0 = SL Hit, -1 = No Trigger
    """
    df = df.copy()
    df["result"] = -1

    for i in range(len(df)):
        entry = df.loc[i, entry_price_col]
        high = df.loc[i, "high"]
        low = df.loc[i, "low"]

        if high >= entry * (1 + target):
            df.loc[i, "result"] = 1  # Take Profit hit
        elif low <= entry * (1 - stoploss):
            df.loc[i, "result"] = 0  # Stop Loss hit

    return df[["timestamp", "entry_price", "high", "low", "exit_price", "result"]]
