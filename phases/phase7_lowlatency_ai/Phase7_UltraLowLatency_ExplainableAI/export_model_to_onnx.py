
import torch
from lstm_transformer_model import LSTMTransformerModel

model = LSTMTransformerModel(input_size=12)
model.load_state_dict(torch.load('lstm_transformer_trading_model.pth'))
model.eval()

dummy_input = torch.randn(1, 50, 12)
torch.onnx.export(
    model, dummy_input, 'trading_model.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch', 1: 'seq'}, 'output': {0: 'batch'}},
    opset_version=12
)
