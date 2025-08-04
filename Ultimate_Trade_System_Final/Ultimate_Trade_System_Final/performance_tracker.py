
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def log_model_performance(y_true, y_pred, model_version, excel_path="model_performance_log.xlsx"):
    results = {
        "Model Version": model_version,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "F1 Score": round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4)
    }

    df_new = pd.DataFrame([results])
    if os.path.exists(excel_path):
        df_old = pd.read_excel(excel_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_excel(excel_path, index=False)
    print("✅ Model performance logged.")

# Usage:
# log_model_performance(y_true=[1,0,1], y_pred=[1,0,0], model_version="20240601")
