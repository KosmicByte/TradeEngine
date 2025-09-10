
from src.tradesystem.utils.auto_label_logger import label_past_trades
from auto_retrain_model import retrain_and_save_model
from src.tradesystem.utils.performance_tracker import log_model_performance
import pandas as pd
import os

def run_full_auto_learning_pipeline(data_path="my_recent_trades.csv"):
    # Step 1: Label trade outcomes
    df = pd.read_csv(data_path)
    labeled_df = label_past_trades(df)
    labeled_df.to_csv("labeled_recent_trades.csv", index=False)
    print("✅ Trades labeled and saved.")

    # Step 2: Retrain model
    model_path = retrain_and_save_model("labeled_recent_trades.csv")

    # Step 3: Mock predictions, for example
    y_true = labeled_df["result"].tolist()
    y_pred = y_true.copy()  # For now assuming perfect match
    model_version = os.path.basename(model_path).replace(".pth", "")

    # Step 4: Log model performance
    log_model_performance(y_true, y_pred, model_version)
    print("🎯 Auto-learning pipeline completed.")

# Example:
# run_full_auto_learning_pipeline("my_recent_trades.csv")
