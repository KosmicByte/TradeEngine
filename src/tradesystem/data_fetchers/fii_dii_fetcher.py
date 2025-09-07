import time
import requests

NSE_HOME = "https://www.nseindia.com/"
NSE_FIIDII_PAGE = "https://www.nseindia.com/market-data/fiidii-market-statistics"
NSE_FIIDII_API = "https://www.nseindia.com/api/fiidiiTradeStatistics?category=all"

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

BASE_HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NSE_FIIDII_PAGE,
    "Connection": "keep-alive",
    "Origin": "https://www.nseindia.com",
}

def _prime_cookies(session: requests.Session) -> None:
    # Touch home + fiidii page to receive ak_bmsc/bm_sv cookies
    for url in (NSE_HOME, NSE_FIIDII_PAGE):
        try:
            session.get(url, timeout=10)
            time.sleep(0.8)  # tiny delay helps with akamai
        except requests.RequestException:
            pass

def _to_float(x):
    try:
        if isinstance(x, str):
            return float(x.replace(",", "").strip())
        return float(x)
    except Exception:
        return 0.0

def fetch_fii_dii_activity(max_retries: int = 2):
    with requests.Session() as s:
        s.headers.update(BASE_HEADERS)
        _prime_cookies(s)

        for attempt in range(1, max_retries + 1):
            try:
                r = s.get(NSE_FIIDII_API, timeout=15)
                if r.status_code == 200 and r.text.strip():
                    data = r.json()
                    break
                # If blocked, re-prime cookies and retry once
                _prime_cookies(s)
                time.sleep(1.0 * attempt)
            except requests.RequestException:
                _prime_cookies(s)
                time.sleep(1.0 * attempt)
        else:
            print("Failed to fetch data from NSE after retries.")
            return "Error"

    rows = data.get("data", []) or []

    net_fii = 0.0
    net_dii = 0.0
    for item in rows:
        cat = (item.get("category") or "").upper()
        net = _to_float(item.get("net", 0))
        if cat == "FII" or cat == "FPI":
            net_fii += net
        elif cat == "DII":
            net_dii += net

    # Return a compact sentiment string + numbers
    sentiment = "Neutral"
    if net_fii - net_dii > 0:
        sentiment = "Bullish (FII net > DII net)"
    elif net_fii - net_dii < 0:
        sentiment = "Bearish (DII net > FII net)"

    return {
        "sentiment": sentiment,
        "net_fii": net_fii,
        "net_dii": net_dii,
        "spread": net_fii - net_dii,
    }

# Example
if __name__ == "__main__":
    result = fetch_fii_dii_activity()
    print("FII/DII Market Sentiment:", result)