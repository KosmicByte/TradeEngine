"""
Pulls FII/DII flows, India VIX, NIFTY spot, and nearest-expiry option chain.
Computes near-ATM PCR (±3%) and a simple market regime label. Saves outputs to
CSV/JSON and prints a concise console summary.

Outputs:
    fii_dii.csv
    nifty_option_chain_nearest_expiry.csv
    snapshot.json
    snapshot.txt
    raw_option_chain_response.json (for debugging)

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
import requests
import sys, signal, random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pytz  # <<< MODIFIED: Added for timezone-aware time checking

# -----------------------------
# Config & constants
# -----------------------------

# --- Path Configuration ---
# Set this number to how many folders "up" from the script is your project root.
LEVELS_UP_TO_PROJECT_ROOT = 3

PROJECT_ROOT = Path(__file__).resolve().parents[LEVELS_UP_TO_PROJECT_ROOT]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Create a unique directory for this specific run using a timestamp
DATE_DIR = RESULTS_DIR / dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATE_DIR.mkdir(parents=True, exist_ok=True)

NSE_BASE = "https://www.nseindia.com"

# Replace NSE_HEADERS with:
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36" # <<< MODIFIED: More common User-Agent
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Referer": "https://www.nseindia.com/option-chain",
    "Connection": "keep-alive",
    "DNT": "1",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "Sec-Ch-Ua": '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
}

# Snappy networking defaults
TIMEOUT = 8
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5

# TCP/HTTP-level retry config
HTTP_TOTAL_RETRIES = 4
HTTP_CONNECT_RETRIES = 3
HTTP_READ_RETRIES = 3
HTTP_BACKOFF_FACTOR = 0.6
STATUS_FORCELIST = [401, 403, 408, 425, 429, 500, 502, 503, 504]

# Hard cap for entire run
RUN_TIMEOUT_SECS = 60

# Endpoints
URL_ALL_INDICES = NSE_BASE + "/api/allIndices"
URL_FIIDII = NSE_BASE + "/api/fiidiiTradeReact"
URL_OPTION_CHAIN = NSE_BASE + "/api/option-chain-indices?symbol={symbol}"

# -----------------------------
# <<< MODIFIED: New function to check NSE market hours
# -----------------------------
def is_nse_market_open() -> bool:
    """Checks if the current time is within NSE trading hours (9:15 AM - 3:30 PM IST)."""
    try:
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = dt.datetime.now(ist)

        # Check if it's a weekday (Monday=0, Sunday=6)
        if now_ist.weekday() > 4:
            return False

        market_open = dt.time(9, 15)
        market_close = dt.time(15, 30)

        return market_open <= now_ist.time() <= market_close
    except Exception as e:
        print(f"[WARN] Could not determine market hours: {e}")
        return True # Default to true to allow running anyway

# -----------------------------
# Global timeout handler
# -----------------------------
def _alarm_handler(signum, frame):
    raise TimeoutError("Global run timeout exceeded")

# -----------------------------
# Helpers
# -----------------------------
def _to_float(x: Any) -> float:
    if x is None: return 0.0
    try:
        if isinstance(x, str) and x.strip() == "": return 0.0
        v = float(x)
        return 0.0 if math.isnan(v) else v
    except Exception: return 0.0

def _normalize_date(s: Any) -> str:
    if s is None: return ""
    if isinstance(s, dt.date): return s.isoformat()
    if isinstance(s, (int, float)):
        try: return dt.date.fromtimestamp(float(s)).isoformat()
        except Exception: return str(s)
    st = str(s).strip()
    for fmt in ("%d-%b-%Y", "%d %b %Y", "%Y-%m-%d", "%d-%B-%Y", "%d %B %Y"):
        try: return dt.datetime.strptime(st, fmt).date().isoformat()
        except Exception: continue
    return st

def _norm_name(s: Any) -> str:
    t = str(s or "").replace("\xa0", " ")
    return " ".join(t.split()).strip().upper()

def _jitter_sleep(base=0.25, spread=0.4):
    time.sleep(base + random.random() * spread)

# -----------------------------
# HTTP session & raw GET with manual retry
# -----------------------------
def _build_session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update(NSE_HEADERS)
    retry = Retry(
        total=HTTP_TOTAL_RETRIES, connect=HTTP_CONNECT_RETRIES, read=HTTP_READ_RETRIES,
        backoff_factor=HTTP_BACKOFF_FACTOR, status_forcelist=STATUS_FORCELIST,
        allowed_methods=frozenset(["GET"]), raise_on_status=False, respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess

def _looks_blocked(text: str) -> bool:
    t = (text or "").lower()
    return ("are you a human" in t) or ("access denied" in t) or ("captcha" in t)

def _retry_get(url: str, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    sess = session or _build_session()
    for i in range(MAX_RETRIES):
        try:
            r = sess.get(url, headers=NSE_HEADERS, timeout=TIMEOUT, allow_redirects=True)

            # <<< MODIFIED: Added detailed logging for diagnostics >>>
            print(f"DEBUG: GET {url} | Status: {r.status_code} | Attempt: {i+1}/{MAX_RETRIES}")
            if r.status_code != 200:
                # Log the beginning of the response to see if it's an HTML block page
                print(f"DEBUG: Response Text (first 200 chars): {r.text[:200].strip()}")
            # <<< END MODIFICATION >>>

            if r.status_code == 200:
                if _looks_blocked(r.text):
                    print("WARN: NSE block page detected, re-warming session...")
                    nse_warmup(sess)
                    time.sleep(RETRY_BACKOFF ** i)
                    continue
                try:
                    js = r.json()
                    return js if isinstance(js, dict) else {"data": js}
                except Exception as e:
                    print(f"WARN: JSON parsing failed on attempt {i+1}. Error: {e}")
                    pass

            if r.status_code in STATUS_FORCELIST:
                print(f"WARN: Retrying due to status {r.status_code}...")
                nse_warmup(sess)
                time.sleep(RETRY_BACKOFF ** i)
                continue

            break
        except requests.exceptions.RequestException as e:
            print(f"WARN: Request failed on attempt {i+1}. Error: {e}")
            time.sleep(RETRY_BACKOFF ** i)
            continue
    print(f"ERROR: Failed to fetch {url} after {MAX_RETRIES} attempts.")
    return {}

# -----------------------------
# Warm-up (cookie priming)
# -----------------------------
def nse_warmup(session: requests.Session) -> None:
    print("INFO: Warming up NSE session to get cookies...")
    try:
        session.get(NSE_BASE + "/", headers=NSE_HEADERS, timeout=TIMEOUT, allow_redirects=True)
        _jitter_sleep(0.3, 0.4)
        session.get(NSE_BASE + "/market-data", headers=NSE_HEADERS, timeout=TIMEOUT, allow_redirects=True)
        _jitter_sleep(0.3, 0.4)
        session.get(NSE_BASE + "/option-chain", headers=NSE_HEADERS, timeout=TIMEOUT, allow_redirects=True)
        _jitter_sleep(0.3, 0.4)
        session.get(URL_ALL_INDICES, headers=NSE_HEADERS, timeout=TIMEOUT, allow_redirects=True)
        print("INFO: Session warmed up.")
    except Exception as e:
        print(f"WARN: Error during session warm-up: {e}")
        pass

# -----------------------------
# Expiry handling
# -----------------------------
def _nearest_expiry(expiry_dates: List[str]) -> Optional[str]:
    if not expiry_dates: return None
    today = dt.date.today()
    def _parse(d: str) -> Optional[dt.date]:
        for fmt in ("%d-%b-%Y", "%d %b %Y", "%d-%B-%Y", "%d %B %Y", "%Y-%m-%d"):
            try: return dt.datetime.strptime(d.strip(), fmt).date()
            except Exception: continue
        return None
    parsed = sorted([t for t in ((_parse(raw), raw) for raw in expiry_dates) if t[0] is not None])
    if not parsed: return None
    for d, raw in parsed:
        if d >= today: return raw
    return parsed[-1][1]

# -----------------------------
# Fetchers
# -----------------------------
def fetch_all_indices(session: Optional[requests.Session] = None) -> Dict[str, Any]:
    return _retry_get(URL_ALL_INDICES, session)

def fetch_fiidii(session: Optional[requests.Session] = None) -> pd.DataFrame:
    js = _retry_get(URL_FIIDII, session)
    data = js.get("data") or js.get("records") or js.get("fiiDii") or []
    if not data: return pd.DataFrame(columns=["category", "date", "buyValue", "sellValue", "netValue"])
    df = pd.DataFrame.from_records(data)
    rename = {
        "category": "category", "Category": "category", "date": "date", "Date": "date", "tradeDate": "date",
        "buyValue": "buyValue", "BuyValue": "buyValue", "buy": "buyValue", "sellValue": "sellValue",
        "SellValue": "sellValue", "sell": "sellValue", "netValue": "netValue", "NetValue": "netValue", "net": "netValue",
    }
    df = df.rename(columns=lambda c: rename.get(c, c))
    for c in ("buyValue", "sellValue", "netValue"):
        if c in df.columns: df[c] = df[c].map(_to_float)
    if "date" in df.columns: df["date"] = df["date"].map(_normalize_date)
    keep = [c for c in ["category", "date", "buyValue", "sellValue", "netValue"] if c in df.columns]
    return df[keep]

def fetch_index_row(indices_json: Dict[str, Any], want_name: str) -> Optional[Dict[str, Any]]:
    want = _norm_name(want_name)
    for row in (indices_json.get("data", []) or []):
        if want in _norm_name(row.get("index", "")): return row
    return None

def fetch_india_vix(session: Optional[requests.Session] = None) -> float:
    idx = fetch_all_indices(session)
    row = fetch_index_row(idx, "INDIA VIX")
    if row:
        for key in ("last", "ltp", "lastPrice"):
            v = _to_float(row.get(key))
            if v > 0: return v
    oc = _retry_get(URL_OPTION_CHAIN.format(symbol="INDIAVIX"), session)
    uv = ((oc or {}).get("records", {}) or {}).get("underlyingValue")
    return float(uv) if uv is not None else float("nan")

def fetch_nifty_spot(session: Optional[requests.Session] = None) -> Optional[float]:
    oc = _retry_get(URL_OPTION_CHAIN.format(symbol="NIFTY"), session)
    uv = ((oc or {}).get("records", {}) or {}).get("underlyingValue")
    if uv is not None:
        v = _to_float(uv)
        if v > 0: return v
    idx = fetch_all_indices(session)
    row = fetch_index_row(idx, "NIFTY 50")
    if row:
        for key in ("last", "ltp", "lastPrice"):
            v = _to_float(row.get(key))
            if v > 0: return v
    return None

def fetch_nifty_option_chain_nearest_expiry(session: Optional[requests.Session] = None) -> pd.DataFrame:
    oc = _retry_get(URL_OPTION_CHAIN.format(symbol="NIFTY"), session)

    # <<< MODIFIED: Save raw JSON response for debugging purposes >>>
    raw_json_path = DATE_DIR / "raw_option_chain_response.json"
    try:
        with open(raw_json_path, "w", encoding="utf-8") as f:
            json.dump(oc, f, indent=2, ensure_ascii=False)
        print(f"INFO: Raw option chain JSON saved to {raw_json_path}")
    except Exception as e:
        print(f"WARN: Could not save raw option chain JSON: {e}")
    # <<< END MODIFICATION >>>

    rec = (oc or {}).get("records", {}) or {}
    rows = rec.get("data", []) or []
    expiries = [str(x) for x in (rec.get("expiryDates", []) or [])]

    def _aggregate(_rows: List[dict]) -> pd.DataFrame:
        agg: Dict[float, Dict[str, float]] = {}
        for r in _rows:
            k = _to_float(r.get("strikePrice"))
            if k <= 0: continue
            ce, pe = r.get("CE", {}), r.get("PE", {})
            e = agg.setdefault(k, {"strike": k, "ce_oi": 0.0, "ce_chg_oi": 0.0, "ce_vol": 0.0, "ce_iv": 0.0,
                                   "pe_oi": 0.0, "pe_chg_oi": 0.0, "pe_vol": 0.0, "pe_iv": 0.0,})
            if ce:
                e["ce_oi"] += _to_float(ce.get("openInterest"))
                e["ce_chg_oi"] += _to_float(ce.get("changeinOpenInterest"))
                e["ce_vol"] += _to_float(ce.get("totalTradedVolume"))
                if _to_float(ce.get("impliedVolatility")) > 0: e["ce_iv"] = _to_float(ce.get("impliedVolatility"))
            if pe:
                e["pe_oi"] += _to_float(pe.get("openInterest"))
                e["pe_chg_oi"] += _to_float(pe.get("changeinOpenInterest"))
                e["pe_vol"] += _to_float(pe.get("totalTradedVolume"))
                if _to_float(pe.get("impliedVolatility")) > 0: e["pe_iv"] = _to_float(pe.get("impliedVolatility"))
        cols = ["strike","ce_oi","ce_chg_oi","ce_vol","ce_iv","pe_oi","pe_chg_oi","pe_vol","pe_iv"]
        return pd.DataFrame(list(agg.values()), columns=cols).sort_values("strike").reset_index(drop=True) if agg else pd.DataFrame(columns=cols)

    nearest = _nearest_expiry(expiries) if expiries else None
    filt = [r for r in rows if str(r.get("expiryDate","")).strip() == nearest] if nearest else []
    df = _aggregate(filt)
    if not df.empty: return df

    if rows:
        buckets = {}
        for r in rows: buckets.setdefault(str(r.get("expiryDate","")).strip(), []).append(r)
        if buckets:
            best_expiry = max(buckets.items(), key=lambda kv: len(kv[1]))[0]
            df = _aggregate(buckets[best_expiry])
            if not df.empty: return df
    return _aggregate(rows)

# -----------------------------
# Metrics & labels
# -----------------------------
def compute_pcr_near_atm(chain: pd.DataFrame, spot: float, pct_window: float = 0.03) -> float:
    if chain.empty or not math.isfinite(spot) or spot <= 0: return 1.0
    lower, upper = spot * (1 - pct_window), spot * (1 + pct_window)
    sub = chain[(chain["strike"] >= lower) & (chain["strike"] <= upper)]
    if sub.empty: sub = chain
    ce_oi, pe_oi = float(sub["ce_oi"].sum()), float(sub["pe_oi"].sum())
    return (pe_oi / ce_oi) if ce_oi > 0 else 1.0

def market_sentiment_from_pcr(pcr: float) -> str:
    if not math.isfinite(pcr): return "Neutral"
    if pcr < 0.8: return "Bearish"
    if pcr > 1.2: return "Bullish"
    return "Neutral"

def market_regime_from_vix(vix: float, baseline: float = 12.0) -> str:
    if not math.isfinite(vix) or vix <= 0: return "Unknown"
    if vix < max(10.0, baseline * 0.9): return "Calm"
    if vix > max(16.0, baseline * 1.2): return "Stressed"
    return "Normal"

# -----------------------------
# Attempt + healing passes
# -----------------------------
def _attempt_snapshot(sess: requests.Session) -> dict:
    nse_warmup(sess); _jitter_sleep()
    fii = fetch_fiidii(sess); _jitter_sleep()
    oc  = fetch_nifty_option_chain_nearest_expiry(sess); _jitter_sleep()
    spot = fetch_nifty_spot(sess) or float("nan"); _jitter_sleep()
    vix  = fetch_india_vix(sess); _jitter_sleep()
    pcr = compute_pcr_near_atm(oc, spot, 0.03)
    return {
        "fii": fii, "oc": oc, "spot": spot, "vix": vix, "pcr": pcr,
        "sentiment": market_sentiment_from_pcr(pcr), "regime": market_regime_from_vix(vix),
        "ok_fii": True, "ok_oc": not oc.empty,
        "ok_spot": math.isfinite(spot), "ok_vix": math.isfinite(vix),
    }

def _heal_missing(sess: requests.Session, snap: dict) -> dict:
    if not snap["ok_oc"]:
        print("INFO: Healing pass: refetching option chain...")
        nse_warmup(sess); _jitter_sleep()
        oc = fetch_nifty_option_chain_nearest_expiry(sess)
        if not oc.empty:
            snap.update({"oc": oc, "ok_oc": True})
            if snap["ok_spot"]:
                snap["pcr"] = compute_pcr_near_atm(oc, snap["spot"])
                snap["sentiment"] = market_sentiment_from_pcr(snap["pcr"])
    if not snap["ok_spot"]:
        print("INFO: Healing pass: refetching NIFTY spot...")
        nse_warmup(sess); _jitter_sleep()
        spot = fetch_nifty_spot(sess) or float("nan")
        snap.update({"spot": spot, "ok_spot": math.isfinite(spot)})
        if snap["ok_spot"] and snap["ok_oc"]:
            snap["pcr"] = compute_pcr_near_atm(snap["oc"], spot)
            snap["sentiment"] = market_sentiment_from_pcr(snap["pcr"])
    if not snap["ok_vix"]:
        print("INFO: Healing pass: refetching India VIX...")
        nse_warmup(sess); _jitter_sleep()
        vix = fetch_india_vix(sess)
        snap.update({"vix": vix, "ok_vix": math.isfinite(vix), "regime": market_regime_from_vix(vix)})
    return snap

# -----------------------------
# Main orchestration
# -----------------------------
def main() -> None:
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(RUN_TIMEOUT_SECS)

    # <<< MODIFIED: Added market hours check at the beginning >>>
    if not is_nse_market_open():
        print("\n[WARNING] 🕒 NSE market is currently closed. Live data like spot price and option chain may be stale or unavailable.\n")
    # <<< END MODIFICATION >>>

    sess = _build_session()
    snapshot = None
    for i in range(1, 4):
        print(f"\n--- Starting fetch attempt {i}/3 ---")
        snap = _attempt_snapshot(sess)
        if snap["ok_oc"] and snap["ok_spot"] and snap["ok_vix"]:
            snapshot = snap
            print("--- Fetch successful on first try! ---")
            break
        print("--- Initial fetch incomplete, attempting to heal missing data... ---")
        snap = _heal_missing(sess, snap)
        if snap["ok_oc"] and snap["ok_spot"] and snap["ok_vix"]:
            snapshot = snap
            print("--- Fetch successful after healing pass! ---")
            break
        _jitter_sleep(0.5, 0.8)

    if snapshot is None:
        print("\n[FATAL] Could not retrieve complete data after multiple attempts. Using best-effort data.")
        snapshot = snap

    # Unpack for saving/printing
    fii_dii_df, oc_df = snapshot["fii"], snapshot["oc"]
    nifty_spot, india_vix = snapshot["spot"], snapshot["vix"]
    pcr, sentiment, regime = snapshot["pcr"], snapshot["sentiment"], snapshot["regime"]

    # Save outputs
    fii_dii_csv = DATE_DIR / "fii_dii.csv"
    oc_csv = DATE_DIR / "nifty_option_chain_nearest_expiry.csv"
    snapshot_json = DATE_DIR / "snapshot.json"
    snapshot_txt = DATE_DIR / "snapshot.txt"

    fii_dii_df.to_csv(fii_dii_csv, index=False)
    oc_df.to_csv(oc_csv, index=False)

    snapshot_meta = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "nifty_spot": round(float(nifty_spot), 2) if math.isfinite(nifty_spot) else None,
        "india_vix": round(float(india_vix), 2) if math.isfinite(india_vix) else None,
        "pcr_near_atm": round(float(pcr), 3) if math.isfinite(pcr) else None,
        "market_sentiment": sentiment,
        "market_regime": regime,
        "files": {"fii_dii_csv": str(fii_dii_csv), "option_chain_csv": str(oc_csv)},
    }
    with open(snapshot_json, "w", encoding="utf-8") as f:
        json.dump(snapshot_meta, f, ensure_ascii=False, indent=2)

    # Human-friendly text snapshot
    lines = [
        "--- Market Snapshot ---",
        f"NIFTY Spot: {nifty_spot if math.isfinite(nifty_spot) else 'FAILED TO FETCH'}",
        f"India VIX:  {india_vix if math.isfinite(india_vix) else 'FAILED TO FETCH'}",
        f"PCR (±3%):  {pcr:.3f}" if math.isfinite(pcr) else "PCR: N/A",
        f"Sentiment:  {sentiment}",
        f"Regime:     {regime}",
        "\nFII/DII Data (head):",
        fii_dii_df.head(6).to_string(index=False) if not fii_dii_df.empty else "  <no data>",
        "\nNIFTY Option Chain — Nearest Expiry (head):",
        oc_df.head(10).to_string(index=False) if not oc_df.empty else "  <no data>"
    ]
    text = "\n".join(lines)
    with open(snapshot_txt, "w", encoding="utf-8") as f: f.write(text + "\n")

    print("\n" + text)
    print("\n✅ Saved outputs to:")
    print(f" - {fii_dii_csv}")
    print(f" - {oc_csv}")
    print(f" - {snapshot_json}")
    print(f" - {snapshot_txt}")

    signal.alarm(0)

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except TimeoutError as e:
        print(f"\n[FATAL] {e}")
        sys.exit(124)
    except Exception as e:
        print(f"\n[FATAL] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)