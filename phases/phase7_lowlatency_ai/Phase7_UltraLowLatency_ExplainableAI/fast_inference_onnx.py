
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('trading_model.onnx', providers=['CUDAExecutionProvider'])

def predict(features_array):
    inputs = {session.get_inputs()[0].name: features_array.astype(np.float32)}
    outputs = session.run(None, inputs)
    pred = np.argmax(outputs[0], axis=1)[0]
    return ['Buy', 'Sell', 'Hold'][pred]
