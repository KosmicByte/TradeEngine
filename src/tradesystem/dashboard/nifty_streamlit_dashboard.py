
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from io import BytesIO

st.set_page_config(layout="wide")
st.title("📊 Nifty | Bank Nifty | Fin Nifty Trading Dashboard")

symbols = {
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK",
    "Fin Nifty": "^CNXFINSERV"
}
selected_symbol = st.selectbox("Choose Index", list(symbols.keys()))
symbol = symbols[selected_symbol]

@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, interval="15m", period="1mo")
    df.dropna(inplace=True)
    df["RSI"] = talib.RSI(df["Close"], timeperiod=14)
    df["MACD"], df["MACD_Signal"], _ = talib.MACD(df["Close"])
    df["EMA20"] = talib.EMA(df["Close"], timeperiod=20)
    df["EMA50"] = talib.EMA(df["Close"], timeperiod=50)
    return df

def backtest(df):
    trades = []
    for i in range(50, len(df)):
        row = df.iloc[i]
        rsi = row["RSI"]
        macd, signal = row["MACD"], row["MACD_Signal"]
        close = row["Close"]
        timestamp = row.name

        if np.isnan([rsi, macd, signal]).any():
            continue

        if rsi < 30 and macd > signal:
            sl = df["Low"].iloc[i-5:i].min()
            target = close + 2 * (close - sl)
            rr = (target - close) / (close - sl)
            trades.append(["BUY", timestamp, close, sl, target, rr])
        elif rsi > 70 and macd < signal:
            sl = df["High"].iloc[i-5:i].max()
            target = close - 2 * (sl - close)
            rr = (close - target) / (sl - close)
            trades.append(["SELL", timestamp, close, sl, target, rr])
    return trades

def suggest_option_leg(signal_type, spot):
    strike = round(spot / 100) * 100
    if signal_type == "BUY":
        return f"Buy CE @ Strike {strike + 100}"
    elif signal_type == "SELL":
        return f"Buy PE @ Strike {strike - 100}"
    return "No suggestion"

def plot_chart(df, trades):
    df = df.copy()
    df["Signal"] = None
    for t in trades:
        df.loc[t[1], "Signal"] = t[0]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Close"], label="Close", color="blue")
    ax.plot(df["EMA20"], label="EMA 20", linestyle="--", color="orange")
    ax.plot(df["EMA50"], label="EMA 50", linestyle="--", color="green")
    buy_signals = df[df["Signal"] == "BUY"]
    sell_signals = df[df["Signal"] == "SELL"]
    ax.scatter(buy_signals.index, buy_signals["Close"], label="BUY", color="lime", marker="^")
    ax.scatter(sell_signals.index, sell_signals["Close"], label="SELL", color="red", marker="v")
    ax.legend()
    ax.set_title("Trade Signals with EMA")
    ax.grid()
    return fig

# Load Data & Trades
df = load_data(symbol)
trades = backtest(df)
valid_trades = [t for t in trades if t[-1] >= 2]

# Display Signals
st.subheader("🔍 Latest Trade Signals")
if valid_trades:
    df_trades = pd.DataFrame(valid_trades, columns=["Signal", "Time", "Entry", "SL", "Target", "RR"])
    df_trades["Option Suggestion"] = df_trades.apply(lambda x: suggest_option_leg(x["Signal"], x["Entry"]), axis=1)
    st.dataframe(df_trades.tail(5), use_container_width=True)

    # Chart
    st.subheader("📈 Price Chart with Trades")
    fig = plot_chart(df, valid_trades)
    st.pyplot(fig)

    # Download Excel
    st.subheader("⬇️ Download Trade Log")
    excel_buffer = BytesIO()
    df_trades.to_excel(excel_buffer, index=False)
    st.download_button("Download Trade Log Excel", data=excel_buffer.getvalue(), file_name="trades.xlsx")

else:
    st.info("No valid trades found with RR ≥ 2 in the recent data.")
