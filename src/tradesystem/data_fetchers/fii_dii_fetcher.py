
import requests
import pandas as pd
from io import StringIO

def fetch_fii_dii_activity():
    """
    Fetches the latest FII and DII trading activity from NSE India and summarizes the net activity.
    Returns: 
        str: A summary of the net FII and DII activity ("Strong Buying", "
                "Strong Selling", or "Mixed").
    """
    url = "https://www.nseindia.com/api/fiidiiTradeStatistics?category=all"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/"
    }

    with requests.Session() as session:
        session.headers.update(headers)
        _ = session.get("https://www.nseindia.com")
        r = session.get(url)

    data = r.json()
    summary = data.get("data", [])

    net_fii = 0
    net_dii = 0
    for item in summary:
        if item["category"] == "FII":
            net_fii = float(item["netValue"].replace(",", ""))
        elif item["category"] == "DII":
            net_dii = float(item["netValue"].replace(",", ""))

    if net_fii > 0 and net_dii > 0:
        return "Strong Buying"
    elif net_fii < 0 and net_dii < 0:
        return "Strong Selling"
    else:
        return "Mixed"
