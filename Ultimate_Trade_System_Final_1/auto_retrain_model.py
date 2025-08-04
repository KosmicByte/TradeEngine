
import pandas as pd
import torch
import os
from train_lstm_model import train_model, TradeDataset
from lstm_transformer_model import LSTMTransformerModel
from trade_dataset import create_labeled_dataset
import datetime

def retrain_and_save_model(data_path="nifty_data_sample.csv", version_folder="model_versions", seq_len=50):
    df = pd.read_csv(data_path)
    dataset = create_labeled_dataset(df, seq_len=seq_len)

    if len(dataset) < 100:
        raise ValueError("Not enough data to retrain model.")

    model = train_model(dataset)

    today = datetime.datetime.now().strftime("%Y%m%d")
    os.makedirs(version_folder, exist_ok=True)
    model_path = os.path.join(version_folder, f"lstm_model_v{today}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model retrained and saved to: {model_path}")
    return model_path
