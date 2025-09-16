"""
Pulls FII/DII flows, India VIX, NIFTY spot, and nearest-expiry option chain.
Computes near-ATM PCR (±3%) and a simple market regime label. Saves outputs to
CSV/JSON and prints a concise console summary.

This version is modified to include anti-bot-blocking measures:
1.  Replaced `requests` with `curl_cffi` to impersonate a real browser fingerprint.
2.  Proxy support via HTTP_PROXY/HTTPS_PROXY environment variables.
3.  Optional browser warm-up with Playwright to harvest valid Akamai cookies.

Author: Abhinav Mishra
"""

from __future__ import annotations

import json
import math
import time
import datetime as dt
from typing import Any, Dict, List, Optional
from pathlib import Path

import pandas as pd
# <<< MODIFIED: Replaced requests with curl_cffi for browser impersonation
from curl_cffi.requests import Session
# <<< END MODIFICATION
import sys, signal, random, os
import pytz

PROXIES = {
    "http":  os.environ.get("HTTP_PROXY")  or os.environ.get("http_proxy"),
    "https": os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"),
}

# -----------------------------
# Config & constants
# -----------------------------
LEVELS_UP_TO_PROJECT_ROOT = 3
PROJECT_ROOT = Path(__file__).resolve().parents[LEVELS_UP_TO_PROJECT_ROOT]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_DIR = RESULTS_DIR / dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)

NSE_BASE = "https://www.nseindia.com"

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Referer": "https://www.nseindia.com/option-chain",
    "Connection": "keep-alive",
}

TIMEOUT = 15
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5
RUN_TIMEOUT_SECS = 90

URL_ALL_INDICES = NSE_BASE + "/api/allIndices"
URL_FIIDII = NSE_BASE + "/api/fiidiiTradeReact"
URL_OPTION_CHAIN = NSE_BASE + "/api/option-chain-indices?symbol={symbol}"

def playwright_cookies_and_ua() -> tuple[dict, str]:
    print("INFO: Launching headless browser to perform warm-up and get valid cookies...")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.nseindia.com/option-chain", wait_until="domcontentloaded")
        time.sleep(2)
        cookies = {c["name"]: c["value"] for c in context.cookies()}
        ua = page.evaluate("() => navigator.userAgent")
        browser.close()
    print("INFO: Browser warm-up complete. Cookies and User-Agent harvested.")
    return cookies, ua

def is_nse_market_open() -> bool:
    try:
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = dt.datetime.now(ist)
        if now_ist.weekday() > 4: return False
        market_open, market_close = dt.time(9, 15), dt.time(15, 30)
        return market_open <= now_ist.time() <= market_close
    except Exception as e:
        print(f"[WARN] Could not determine market hours: {e}")
        return True

def _alarm_handler(signum, frame):
    raise TimeoutError("Global run timeout exceeded")

def _to_float(x: Any) -> float:
    if x is None: return 0.0
    try:
        if isinstance(x, str) and x.strip() in ("", "-"): return 0.0
        v = float(str(x).replace(",", ""))
        return 0.0 if math.isnan(v) else v
    except Exception: return 0.0

def _normalize_date(s: Any) -> str:
    if s is None: return ""
    st = str(s).strip()
    for fmt in ("%d-%b-%Y", "%d %b %Y", "%Y-%m-%d"):
        try: return dt.datetime.strptime(st, fmt).date().isoformat()
        except Exception: continue
    return st

def _norm_name(s: Any) -> str:
    t = str(s or "").replace("\xa0", " ")
    return " ".join(t.split()).strip().upper()

def _jitter_sleep(base=0.25, spread=0.4):
    time.sleep(base + random.random() * spread)

# <<< MODIFIED: _build_session now uses curl_cffi and impersonation
def _build_session() -> Session:
    """Builds a curl_cffi Session that impersonates a Chrome browser."""
    # Proxies are passed directly to the Session constructor
    proxies = {k: v for k, v in PROXIES.items() if v}
    if proxies:
        print(f"INFO: Using proxies: {proxies}")

    # The impersonate parameter is key to defeating bot detection
    sess = Session(
        impersonate="chrome120",
        proxies=proxies if proxies else None,
        timeout=TIMEOUT
    )
    sess.headers.update(NSE_HEADERS)
    return sess
# <<< END MODIFICATION

def _looks_blocked(text: str) -> bool:
    t = (text or "").lower()
    return ("access denied" in t) or ("captcha" in t)

def _retry_get(url: str, session: Session) -> Dict[str, Any]:
    for i in range(MAX_RETRIES):
        try:
            r = session.get(url, headers=NSE_HEADERS)
            print(f"DEBUG: GET {url} | Status: {r.status_code} | Attempt: {i+1}/{MAX_RETRIES}")

            if _looks_blocked(r.text):
                print(f"WARN: Block page detected on attempt {i+1}. Retrying...")
                print(f"DEBUG: Response Text (first 200 chars): {r.text[:200].strip()}")
                _jitter_sleep(RETRY_BACKOFF ** i, 1.0)
                continue

            if r.status_code == 200:
                try:
                    js = r.json()
                    return js if isinstance(js, dict) else {"data": js}
                except Exception as e:
                    print(f"WARN: JSON parsing failed on attempt {i+1}. Error: {e}")

            _jitter_sleep(RETRY_BACKOFF ** i, 1.0)
        except Exception as e:
            print(f"WARN: Request failed on attempt {i+1}. Error: {e}")
            _jitter_sleep(RETRY_BACKOFF ** i, 1.0)

    print(f"ERROR: Failed to fetch {url} after {MAX_RETRIES} attempts.")
    return {}

# The rest of the script remains largely the same, but uses the new session object
def fetch_all_indices(session: Session) -> Dict[str, Any]:
    return _retry_get(URL_ALL_INDICES, session)

def fetch_fiidii(session: Session) -> pd.DataFrame:
    js = _retry_get(URL_FIIDII, session)
    data = js.get("data") or []
    if not data: return pd.DataFrame()
    df = pd.DataFrame.from_records(data)
    df.rename(columns={"netValue": "net", "buyValue": "buy", "sellValue": "sell"}, inplace=True)
    for col in ["buy", "sell", "net"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_float)
    if "date" in df.columns:
        df["date"] = df["date"].apply(_normalize_date)
    return df

def fetch_index_row(indices_json: Dict[str, Any], want_name: str) -> Optional[Dict[str, Any]]:
    want = _norm_name(want_name)
    for row in (indices_json.get("data", []) or []):
        if want in _norm_name(row.get("index", "")): return row
    return None

def fetch_india_vix(session: Session) -> float:
    idx = fetch_all_indices(session)
    if row := fetch_index_row(idx, "INDIA VIX"):
        if (v := _to_float(row.get("last"))) > 0: return v
    return float("nan")

def fetch_nifty_spot(session: Session) -> Optional[float]:
    oc = _retry_get(URL_OPTION_CHAIN.format(symbol="NIFTY"), session)
    if uv := ((oc or {}).get("records", {}) or {}).get("underlyingValue"):
        if (v := _to_float(uv)) > 0: return v
    return None

def fetch_nifty_option_chain_nearest_expiry(session: Session) -> pd.DataFrame:
    oc = _retry_get(URL_OPTION_CHAIN.format(symbol="NIFTY"), session)
    raw_json_path = RUN_DIR / "raw_option_chain_response.json"
    try:
        with open(raw_json_path, "w", encoding="utf-8") as f:
            json.dump(oc, f, indent=2, ensure_ascii=False)
    except Exception: pass

    rec = (oc or {}).get("records", {}) or {}
    rows = rec.get("data", []) or []
    return pd.DataFrame(rows) if rows else pd.DataFrame()

# Analysis functions are unchanged
def compute_pcr_near_atm(chain: pd.DataFrame, spot: float, pct_window: float = 0.03) -> float:
    if chain.empty or not math.isfinite(spot) or spot <= 0: return 1.0
    chain['strikePrice'] = chain['strikePrice'].apply(_to_float)
    sub = chain[(chain["strikePrice"] >= spot * (1-pct_window)) & (chain["strikePrice"] <= spot * (1+pct_window))]
    if sub.empty: sub = chain
    ce_oi = sub['CE'].apply(lambda x: x.get('openInterest', 0) if isinstance(x, dict) else 0).sum()
    pe_oi = sub['PE'].apply(lambda x: x.get('openInterest', 0) if isinstance(x, dict) else 0).sum()
    return (pe_oi / ce_oi) if ce_oi > 0 else 1.0

def market_sentiment_from_pcr(pcr: float) -> str:
    if not math.isfinite(pcr): return "Neutral"
    if pcr < 0.8: return "Bearish"
    if pcr > 1.2: return "Bullish"
    return "Neutral"

def market_regime_from_vix(vix: float, baseline: float = 12.0) -> str:
    if not math.isfinite(vix) or vix <= 0: return "Unknown"
    if vix < baseline * 0.9: return "Calm"
    if vix > baseline * 1.2: return "Stressed"
    return "Normal"

def main() -> None:
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(RUN_TIMEOUT_SECS)

    if not is_nse_market_open():
        print("\n[WARNING] 🕒 NSE market is currently closed. Live data may be stale or unavailable.\n")

    cookies = {}
    try:
        cookies, ua = playwright_cookies_and_ua()
        NSE_HEADERS["User-Agent"] = ua
    except Exception as e:
        print(f"[WARN] Playwright warm-up failed, continuing without it. Error: {e}")

    # Build the impersonating session
    sess = _build_session()

    # <<< MODIFIED: Cookie handling for curl_cffi session
    if cookies:
        print("INFO: Injecting cookies from browser warm-up into the session.")
        for k, v in cookies.items():
            sess.cookies.set(k, v, domain=".nseindia.com", path="/")
    # <<< END MODIFICATION

    print("\n--- Starting Data Fetch ---")
    fii_df = fetch_fiidii(sess)
    oc_df = fetch_nifty_option_chain_nearest_expiry(sess)
    spot = fetch_nifty_spot(sess) or float("nan")
    vix = fetch_india_vix(sess)

    pcr = compute_pcr_near_atm(oc_df, spot)
    sentiment = market_sentiment_from_pcr(pcr)
    regime = market_regime_from_vix(vix)

    # Save outputs
    if not fii_df.empty:
        fii_df.to_csv(RUN_DIR / "fii_dii.csv", index=False)
    if not oc_df.empty:
        oc_df.to_csv(RUN_DIR / "nifty_option_chain_nearest_expiry.csv", index=False)

    snapshot_meta = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "nifty_spot": round(spot, 2) if math.isfinite(spot) else None,
        "india_vix": round(vix, 2) if math.isfinite(vix) else None,
        "pcr_near_atm": round(pcr, 3) if math.isfinite(pcr) else None,
        "market_sentiment": sentiment, "market_regime": regime,
    }
    with open(RUN_DIR / "snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot_meta, f, indent=2)

    # Print summary
    print("\n--- Market Snapshot ---")
    print(f"NIFTY Spot: {spot if math.isfinite(spot) else 'FAILED'}")
    print(f"India VIX:  {vix if math.isfinite(vix) else 'FAILED'}")
    print(f"PCR (±3%):  {pcr:.3f}" if math.isfinite(pcr) else "PCR: N/A")
    print(f"Sentiment:  {sentiment}")
    print(f"Regime:     {regime}")
    print("\n✅ Saved outputs to:")
    print(f" - {RUN_DIR}")

    signal.alarm(0)

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)