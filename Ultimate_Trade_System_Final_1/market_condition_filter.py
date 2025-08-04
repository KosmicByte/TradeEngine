
from vix_volatility_filter import fetch_nifty_vix, atr_trend_filter
from news_sentiment_fetcher import fetch_market_sentiment
from fii_dii_fetcher import fetch_fii_dii_activity
import pandas as pd

def evaluate_market_condition(df_ohlcv):
    try:
        # VIX
        vix = fetch_nifty_vix()
        vix_status = "High Vol" if vix > 17 else "Stable"

        # ATR Trend
        atr_trend, atr_data = atr_trend_filter(df_ohlcv)

        # Sentiment
        sentiment = fetch_market_sentiment()

        # FII/DII Flow
        flow = fetch_fii_dii_activity()

        # Combine
        if sentiment == "Negative" or vix_status == "High Vol" or atr_trend == "Volatile" or flow == "Strong Selling":
            market_condition = "AVOID TRADING"
        elif sentiment == "Positive" and flow == "Strong Buying" and atr_trend == "Calm":
            market_condition = "FAVORABLE"
        else:
            market_condition = "NEUTRAL / RANGE"

        return {
            "vix": vix,
            "vix_status": vix_status,
            "atr_trend": atr_trend,
            "sentiment": sentiment,
            "fii_dii": flow,
            "regime": market_condition
        }

    except Exception as e:
        print("Market filter failed:", e)
        return {"regime": "Unknown"}
