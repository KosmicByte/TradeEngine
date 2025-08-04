
import pandas as pd

def detect_BOS_CHoCH(df):
    df['BOS'] = (df['high'] > df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
    df['CHoCH'] = (df['high'] < df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
    return df

# Example DataFrame format:
# df = pd.DataFrame({ 'high': [...], 'low': [...] })
# df = detect_BOS_CHoCH(df)
