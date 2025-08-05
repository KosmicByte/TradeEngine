
import pandas as pd
from dynamic_strategy_selector import dynamic_strategy_selector
from strategy_payoff_visualizer import plot_option_strategy

def integrated_dynamic_strategy_engine(data_path, symbol="NIFTY", index=True, lot_size=50):
    """
    Runs the full dynamic strategy selector and visualizes the best strategy payoff.
    """
    # Load OHLCV data
    df = pd.read_csv(data_path)

    # Get dynamic strategy result
    result = dynamic_strategy_selector(df, symbol=symbol, index=index, lot_size=lot_size)

    print("🔍 Dynamic Strategy Engine Result:")
    for key, value in result.items():
        print(f"{key}: {value}")

    # If trading decision, plot payoff
    if result.get("decision") == "TRADE":
        strat = result["best_strategy"]
        legs = strat["legs"]
        # Use actual spot price
        spot_price = df.iloc[-1]["close"]
        filename = f"dynamic_{symbol}_payoff.png"
        plot_option_strategy(strat["strategy"], legs, spot_price=spot_price, save_path=filename)
        print(f"✅ Payoff chart saved to {filename}")
    else:
        print("No trade to execute based on dynamic strategy selector.")

# Example usage:
# integrated_dynamic_strategy_engine("latest_15min_candles.csv", symbol="BANKNIFTY")
