
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lstm_transformer_model import LSTMTransformerModel
from src.tradesystem.feature_engineering.trade_dataset import TradeDataset
import pandas as pd
from src.tradesystem.feature_engineering.feature_engineering import add_features

# Load and preprocess data
df = pd.read_csv("nifty_data_sample.csv")
df = add_features(df)
df = df.iloc[:-10]
from src.tradesystem.feature_engineering.trade_dataset import label_trades
df["label"] = label_trades(df)

# Define dataset and dataloader
dataset = TradeDataset(df)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
input_size = dataset[0][0].shape[1]
model = LSTMTransformerModel(input_size=input_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # adjust as needed
    model.train()
    running_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(loader):.4f}")

# Save model
torch.save(model.state_dict(), "lstm_transformer_trading_model.pth")
