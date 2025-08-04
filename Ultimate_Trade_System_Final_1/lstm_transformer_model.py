
import torch
import torch.nn as nn

class LSTMTransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3, dropout=0.3):
        super(LSTMTransformerModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(hidden_size*2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(64, output_size)  # Output: Buy, Sell, Hold

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = torch.mean(attn_output, dim=1)
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc_out(x)
        return output

# Example:
# model = LSTMTransformerModel(input_size=12, output_size=3)
# x = torch.randn(32, 50, 12)  # batch_size, sequence_length, features
# out = model(x)
# print(out.shape)  # [32, 3]
