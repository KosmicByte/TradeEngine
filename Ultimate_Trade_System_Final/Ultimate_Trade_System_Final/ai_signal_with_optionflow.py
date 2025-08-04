
from predict_signal_realtime import predict_trade_signal
from nse_option_chain_fetcher import fetch_nse_option_chain
from option_flow_analyzer import analyze_option_flow
import pandas as pd

def generate_final_trade_signal(df_ohlcv):
    # AI Prediction
    ai_signal = predict_trade_signal(df_ohlcv)

    # Option Bias Confirmation
    try:
        df_option = fetch_nse_option_chain()
        summary, _ = analyze_option_flow(df_option)
        option_bias = summary["bias"]
    except Exception as e:
        print("Option flow fetch failed:", e)
        option_bias = "Neutral"

    # Final Decision Logic
    if ai_signal == "Buy" and option_bias == "Bullish":
        return "CONFIRMED BUY"
    elif ai_signal == "Sell" and option_bias == "Bearish":
        return "CONFIRMED SELL"
    elif ai_signal == "Hold":
        return "HOLD"
    else:
        return f"WEAK SIGNAL ({ai_signal} vs {option_bias})"

# Example:
# df = pd.read_csv("latest_15min_candles.csv")
# print(generate_final_trade_signal(df))
