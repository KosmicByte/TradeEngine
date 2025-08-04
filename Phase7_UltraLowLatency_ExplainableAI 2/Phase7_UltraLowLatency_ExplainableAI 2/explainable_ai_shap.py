
import shap
import numpy as np
import redis
import json
from onnxruntime import InferenceSession

session = InferenceSession('trading_model.onnx', providers=['CPUExecutionProvider'])
features = json.loads(redis.Redis().get('latest_features') or '{}')
feature_names = list(features.keys())
feature_values = np.array([list(features.values())], dtype=np.float32)

explainer = shap.DeepExplainer(session, feature_values)
shap_vals = explainer.shap_values(feature_values)
importance = dict(zip(feature_names, shap_vals[0][0]))
print("SHAP feature importance:", importance)
