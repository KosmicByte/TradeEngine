
from predict_signal_realtime import predict_trade_signal
from src.tradesystem.data_fetchers.nse_option_chain_fetcher import fetch_nse_option_chain
from src.tradesystem.filters.option_flow_analyzer import analyze_option_flow
from src.tradesystem.filters.market_condition_filter import evaluate_market_condition


def full_trade_decision(df_ohlcv):
    try:
        # Market Regime Filter
        regime_info = evaluate_market_condition(df_ohlcv)
        if regime_info["regime"] == "AVOID TRADING":
            return "BLOCKED: Market Conditions Unfavorable"

        # AI Signal
        ai_signal = predict_trade_signal(df_ohlcv)

        # Option Bias
        df_option = fetch_nse_option_chain()
        summary, _ = analyze_option_flow(df_option)
        option_bias = summary["bias"]

        # Final Decision Logic
        if ai_signal == "Buy" and option_bias == "Bullish":
            return "CONFIRMED BUY"
        elif ai_signal == "Sell" and option_bias == "Bearish":
            return "CONFIRMED SELL"
        elif ai_signal == "Hold":
            return "HOLD"
        else:
            return f"WEAK SIGNAL ({ai_signal} vs {option_bias})"

    except Exception as e:
        return f"ERROR: {e}"
