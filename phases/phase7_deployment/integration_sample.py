import redis
import json
import pandas as pd
from fast_inference_onnx import predict
from explainable_ai_shap import shap_values
from strategy_payoff_visualizer import plot_option_strategy

def integration_loop():
    r = redis.Redis(host='localhost', port=6379, db=0)
    while True:
        # Fetch latest features computed by the pipeline
        features = json.loads(r.get('latest_features') or '{}')
        if not features:
            continue

        # Prepare input array for ONNX model
        import numpy as np
        feature_array = np.array([list(features.values())], dtype=np.float32)
        feature_array = feature_array.reshape(1, -1)  # adjust seq_len if required

        # Predict signal
        signal = predict(feature_array)
        print(f"Predicted Signal: {signal}")

        # Explain features
        importance = shap_values()  # returns dict of feature importance
        print("Feature Importance:", importance)

        time.sleep(1)

if __name__ == "__main__":
    integration_loop()
