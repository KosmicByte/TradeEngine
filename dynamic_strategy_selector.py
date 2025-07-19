
from multi_asset_option_chain import fetch_option_chain
from option_strategy_recommender import suggest_option_strategies
from option_risk_planner import calculate_option_risk
from market_condition_filter import evaluate_market_condition
from final_ai_trade_engine import full_trade_decision

def dynamic_strategy_selector(df_ohlcv, symbol="NIFTY", index=True, lot_size=50):
    """
    Selects the optimal option strategy based on:
      - Market regime (FAVORABLE/NEUTRAL/AVOID TRADING)
      - AI signal (Buy/Sell/Hold)
      - Strategy list (spreads, straddles, condors)
      - Risk:Reward, max profit, complexity penalty
    Returns the best strategy and associated metrics.
    """
    # Market regime
    regime_info = evaluate_market_condition(df_ohlcv)
    if regime_info["regime"] != "FAVORABLE":
        return {"decision": "NO_TRADE", "reason": f"Market {regime_info['regime']}"}

    # AI signal
    signal = full_trade_decision(df_ohlcv)
    if "BUY" in signal.upper():
        bias = "Bullish"
    elif "SELL" in signal.upper():
        bias = "Bearish"
    else:
        bias = "Neutral"

    # Fetch chain and strategies
    df_chain = fetch_option_chain(symbol=symbol, index=index)
    strategies = suggest_option_strategies(df_chain, market_bias=bias)

    # Score strategies
    scored = []
    for strat in strategies:
        legs = strat["legs"]
        # Ensure premiums
        for leg in legs:
            leg.setdefault("premium", 20)
        risk = calculate_option_risk(strat["strategy"], legs, lot_size=lot_size)
        try:
            rr_val = float(risk["rr_ratio"].split(":")[1])
        except:
            rr_val = 0
        profit = risk["max_profit"]
        penalty = len(legs) * 0.1
        score = rr_val * 0.5 + (profit / 1000) * 0.3 - penalty
        scored.append((score, strat, risk))

    if not scored:
        return {"decision": "NO_STRATEGIES", "reason": "No valid strategies"}

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_strat, best_risk = scored[0]
    return {
        "decision": "TRADE",
        "regime": regime_info["regime"],
        "signal": signal,
        "best_strategy": best_strat,
        "risk": best_risk,
        "score": round(best_score, 2)
    }

# Example:
# import pandas as pd
# df = pd.read_csv("latest_15min_candles.csv")
# print(dynamic_strategy_selector(df, symbol="BANKNIFTY"))
