
import pandas as pd
import matplotlib.pyplot as plt

def backtest_strategy(data, rsi_min=30, rsi_max=70, rr_threshold=2.0):
    trades = []
    for i in range(1, len(data)):
        row = data.iloc[i]
        prev = data.iloc[i-1]
        if rsi_min < row['rsi'] < rsi_max and row['signal'] == 'Buy':
            entry = row['close']
            sl = row['low'] - 5
            target = entry + (entry - sl) * rr_threshold
            result = 'Win' if row['high'] >= target else 'Loss'
            trades.append({'entry': entry, 'sl': sl, 'target': target, 'result': result})
    return pd.DataFrame(trades)

def plot_results(results_df):
    win_count = (results_df['result'] == 'Win').sum()
    loss_count = (results_df['result'] == 'Loss').sum()
    plt.bar(['Win', 'Loss'], [win_count, loss_count])
    plt.title("Backtest Outcome")
    plt.show()

# Example usage
# df = pd.read_csv("backtest_data.csv") with 'rsi', 'signal', 'close', 'low', 'high'
# results = backtest_strategy(df)
# plot_results(results)
