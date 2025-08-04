
import requests
import pandas as pd

def fetch_nse_option_chain(symbol="NIFTY"):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain"
    }

    with requests.Session() as session:
        session.headers.update(headers)
        _ = session.get("https://www.nseindia.com")  # Cookie load
        response = session.get(url)

    data = response.json()["records"]["data"]
    records = []
    for item in data:
        strike = item.get("strikePrice")
        ce = item.get("CE", {})
        pe = item.get("PE", {})

        records.append({
            "strike": strike,
            "ce_oi": ce.get("openInterest", 0),
            "ce_chg_oi": ce.get("changeinOpenInterest", 0),
            "ce_vol": ce.get("totalTradedVolume", 0),
            "ce_iv": ce.get("impliedVolatility", 0),
            "pe_oi": pe.get("openInterest", 0),
            "pe_chg_oi": pe.get("changeinOpenInterest", 0),
            "pe_vol": pe.get("totalTradedVolume", 0),
            "pe_iv": pe.get("impliedVolatility", 0)
        })

    df = pd.DataFrame(records)
    df = df.dropna(subset=["strike"])
    return df.sort_values("strike").reset_index(drop=True)

# Example:
# df = fetch_nse_option_chain()
# print(df.head())
