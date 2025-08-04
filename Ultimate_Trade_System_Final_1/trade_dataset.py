
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset

def label_trades(df, atr_col="atr", target_rr=2.0, window=10):
    labels = []
    for i in range(len(df) - window):
        entry = df.iloc[i]["close"]
        sl = entry - df.iloc[i][atr_col]
        tp = entry + df.iloc[i][atr_col] * target_rr
        future_prices = df.iloc[i+1:i+window+1]["close"].values
        if np.any(future_prices >= tp):
            labels.append(0)  # Buy
        elif np.any(future_prices <= sl):
            labels.append(1)  # Sell
        else:
            labels.append(2)  # Hold
    return labels

class TradeDataset(Dataset):
    def __init__(self, df, sequence_length=50):
        self.df = df
        self.seq_len = sequence_length
        self.scaler = MinMaxScaler()
        self.features = self.scaler.fit_transform(df.drop(columns=["label"]))
        self.labels = df["label"].values

    def __len__(self):
        return len(self.df) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_len]
        y = self.labels[idx+self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# Example usage:
# df = add_features(pd.read_csv("nifty_data.csv"))
# df = df.iloc[:-10]  # for labeling
# df["label"] = label_trades(df)
# dataset = TradeDataset(df)
# x, y = dataset[0]
