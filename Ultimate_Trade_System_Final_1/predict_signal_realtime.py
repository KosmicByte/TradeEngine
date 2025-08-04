
import torch
import pandas as pd
from feature_engineering import add_features
from lstm_transformer_model import LSTMTransformerModel
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def prepare_input_sequence(df, seq_len=50):
    df = add_features(df)
    df = df.dropna()
    features = df.drop(columns=["timestamp", "label"], errors="ignore")
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    if len(features_scaled) < seq_len:
        return None
    input_seq = features_scaled[-seq_len:]
    return torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)  # shape: (1, seq_len, input_size)

def predict_trade_signal(df, model_path="lstm_transformer_trading_model.pth"):
    input_seq = prepare_input_sequence(df)
    if input_seq is None:
        return "Insufficient Data"

    input_size = input_seq.shape[2]
    model = LSTMTransformerModel(input_size=input_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    with torch.no_grad():
        output = model(input_seq)
        prediction = torch.argmax(output, dim=1).item()
        if prediction == 0:
            return "Buy"
        elif prediction == 1:
            return "Sell"
        else:
            return "Hold"

# Example:
# df_live = pd.read_csv("latest_15min_candles.csv")
# signal = predict_trade_signal(df_live)
# print("Predicted Signal:", signal)
