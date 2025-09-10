
import pandas as pd
from nsepython import nse_fiidii, nse_optionchain_scrapper, nsefetch
import time

def _retry(fn, *args, retries=3, **kwargs):
    delay = 0.5
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if i == retries-1:
                raise
            time.sleep(delay); delay *= 2

# 1) FII/DII
def fetch_fii_dii() -> pd.DataFrame:
    df = _retry(nse_fiidii)
    # ensure expected columns
    return pd.DataFrame(df)[["category","date","buyValue","sellValue","netValue"]]

# 2) Sentiment from NSE data only
def fetch_market_sentiment() -> str:
    # breadth: use index constituents or a-proxy from nsepython; fallback breadth=1.0 if unavailable
    breadth = 1.0  # TODO: replace with real breadth via nsepython
    # PCR from option chain
    payload = _retry(nse_optionchain_scrapper, "NIFTY")
    data = payload["records"]["data"]
    # compute PCR from summed OI around ATM
    ce_oi = sum([x.get("CE",{}).get("openInterest",0) for x in data])
    pe_oi = sum([x.get("PE",{}).get("openInterest",0) for x in data])
    pcr = (pe_oi / ce_oi) if ce_oi else 1.0
    # VIX change (require previous close if available; else Δ=0)
    vix_now = fetch_nifty_vix()
    vix_prev = vix_now  # TODO: replace with prev if accessible; else 0% change
    vix_delta = 0.0 if vix_prev == 0 else (vix_now - vix_prev) / vix_prev * 100
    # NIFTY return (last vs prevClose if accessible; else 0)
    nifty_ret = 0.0

    score = 0
    score += 1 if breadth > 1.1 else (-1 if breadth < 0.9 else 0)
    score += 0 if 0.9 <= pcr <= 1.1 else (-1 if pcr > 1.1 else 1)
    score += -1 if vix_delta > 5 else (1 if vix_delta < -5 else 0)
    score += 1 if nifty_ret > 0.5 else (-1 if nifty_ret < -0.5 else 0)

    return "Positive" if score > 1 else ("Negative" if score < -1 else "Neutral")

# 3) Option Chain
def fetch_nse_option_chain(symbol: str = "NIFTY") -> pd.DataFrame:
    payload = _retry(nse_optionchain_scrapper, symbol)
    rows = []
    for item in payload["records"]["data"]:
        k = item.get("strikePrice")
        ce = item.get("CE", {})
        pe = item.get("PE", {})
        if k is None:
            continue
        rows.append({
            "strike": k,
            "ce_oi": ce.get("openInterest", 0),
            "ce_chg_oi": ce.get("changeinOpenInterest", 0),
            "ce_vol": ce.get("totalTradedVolume", 0),
            "ce_iv": ce.get("impliedVolatility", 0),
            "pe_oi": pe.get("openInterest", 0),
            "pe_chg_oi": pe.get("changeinOpenInterest", 0),
            "pe_vol": pe.get("totalTradedVolume", 0),
            "pe_iv": pe.get("impliedVolatility", 0),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("strike").reset_index(drop=True)

def fetch_nifty_vix() -> float:
    """
    Returns the latest India VIX value using NSE's allIndices feed via nsepython.nsefetch.
    Robust to minor key/label changes.
    """
    data = nsefetch("https://www.nseindia.com/api/allIndices")  # handled headers/cookies
    items = data.get("data") or data  # some builds return {"data":[...]} others just [...]

    for row in items:
        name = (row.get("index") or row.get("indexName") or "").strip().upper()
        if name in ("INDIA VIX", "INDIAVIX"):
            # Try typical fields in descending likelihood
            for k in ("last", "lastPrice", "lastValue", "value", "ltp"):
                if k in row and row[k] is not None:
                    return float(row[k])
            # Some feeds give 4-decimal fixed-point; keep as-is (human-readable)
            raise KeyError("India VIX found but no usable price field")
    raise KeyError("India VIX not present in allIndices payload")

def atr_trend_filter(df_ohlc: pd.DataFrame, window: int = 14):
    df = df_ohlc.copy()
    df["H-L"] = df["high"] - df["low"]
    df["H-PC"] = (df["high"] - df["close"].shift(1)).abs()
    df["L-PC"] = (df["low"] - df["close"].shift(1)).abs()
    tr = df[["H-L","H-PC","L-PC"]].max(axis=1)
    df["atr"] = tr.rolling(window=window).mean()
    df["atr_slope"] = df["atr"].diff()
    slope = df["atr_slope"].tail(5).mean()
    regime = "Volatile" if slope > 0 else "Calm"
    return regime, df[["atr","atr_slope"]].tail(1)

if __name__ == "__main__":

    # # Printing FII/DII data directly
    # # Printing the data fetched from nse_fiidii calling function
    # print(nse_fiidii())

    print("FII/DII Data:")
    # Printing the data fetched from user's fetch_fii_dii function - essentially a wrapper over nse_fiidii
    print(fetch_fii_dii())

    print("\nMarket Sentiment:")
    print(fetch_market_sentiment())

    print("\nNIFTY Option Chain:")
    print(fetch_nse_option_chain("NIFTY").head())

    print("\nIndia VIX:")
    print(fetch_nifty_vix())