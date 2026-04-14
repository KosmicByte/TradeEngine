# scripts/inspect_params.py
import pickle
path = "params.pkl"
with open(path, "rb") as f:
    p = pickle.load(f)
print(type(p))
if isinstance(p, tuple):
    print(f"Tuple length: {len(p)}")
    for i, item in enumerate(p):
        print(f"  [{i}] type={type(item).__name__}  value={item}")
elif isinstance(p, dict):
    for k, v in p.items():
        print(f"  '{k}': {type(v).__name__}")