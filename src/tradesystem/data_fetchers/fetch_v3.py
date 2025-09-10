#!/usr/bin/env python3
"""
Pulls FII/DII flows, India VIX, NIFTY spot, and nearest-expiry option chain.
Computes near-ATM PCR (±3%) and a simple market regime label. Saves outputs to
CSV/JSON and prints a concise console summary.

Outputs:
    fii_dii.csv
    nifty_option_chain_nearest_expiry.csv
    snapshot.json
    snapshot.txt
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

# -----------------------------
# Config & constants
# -----------------------------

# Where to save outputs: <repo-root>/results
REPO_ROOT = Path(__file__).resolve().parents[3]   # …/repo/src/tradesystem/data_fetchers -> parents[3] = repo root
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATE_DIR = RESULTS_DIR / dt.date.today().isoformat()
DATE_DIR.mkdir(parents=True, exist_ok=True)

NSE_BASE = "https://www.nseindia.com"

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NSE_BASE + "/",
    "Connection": "keep-alive",
    "DNT": "1",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
}

# Snappy networking defaults
TIMEOUT = 8           # per-request timeout seconds
MAX_RETRIES = 3       # manual retries in _retry_get
RETRY_BACKOFF = 1.5   # exponential backoff multiplier

# TCP/HTTP-level retry config
HTTP_TOTAL_RETRIES = 4
HTTP_CONNECT_RETRIES = 3
HTTP_READ_RETRIES = 3
HTTP_BACKOFF_FACTOR = 0.6
STATUS_FORCELIST = [401, 403, 408, 425, 429, 500, 502, 503, 504]

# Hard cap for entire run (prevents “keeps running”)
RUN_TIMEOUT_SECS = 60

# Endpoints
URL_ALL_INDICES = NSE_BASE + "/api/allIndices"
URL_FIIDII = NSE_BASE + "/api/fiidiiTradeReact"
URL_OPTION_CHAIN = NSE_BASE + "/api/option-chain-indices?symbol={symbol}"

# -----------------------------
# Global timeout handler
# -----------------------------
def _alarm_handler(signum, frame):
    raise TimeoutError("Global run timeout exceeded")

# -----------------------------
# Helpers
# -----------------------------
def _to_float(x: Any) -> float:
    """Safer float converter: returns 0.0 for None/NaN/''; passes through floats/ints/strings."""
    if x is None:
        return 0.0
    try:
        if isinstance(x, str) and x.strip() == "":
            return 0.0
        v = float(x)
        if math.isnan(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def _normalize_date(s: Any) -> str:
    """Normalize a date-like value to ISO (YYYY-MM-DD). Returns input string if parse fails."""
    if s is None:
        return ""
    if isinstance(s, dt.date):
        return s.isoformat()
    if isinstance(s, (int, float)):
        try:
            return dt.date.fromtimestamp(float(s)).isoformat()
        except Exception:
            return str(s)
    st = str(s).strip()
    for fmt in ("%d-%b-%Y", "%d %b %Y", "%Y-%m-%d", "%d-%B-%Y", "%d %B %Y"):
        try:
            return dt.datetime.strptime(st, fmt).date().isoformat()
        except Exception:
            continue
    return st


def _norm_name(s: Any) -> str:
    """Compress whitespace (incl. NBSP), strip, and uppercase."""
    t = str(s or "")
    t = t.replace("\xa0", " ")
    t = " ".join(t.split())
    return t.strip().upper()


def _jitter_sleep(base=0.25, spread=0.4):
    time.sleep(base + random.random() * spread)

# -----------------------------
# HTTP session & raw GET with manual retry
# -----------------------------
def _build_session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update(NSE_HEADERS)
    retry = Retry(
        total=HTTP_TOTAL_RETRIES,
        connect=HTTP_CONNECT_RETRIES,
        read=HTTP_READ_RETRIES,
        backoff_factor=HTTP_BACKOFF_FACTOR,
        status_forcelist=STATUS_FORCELIST,
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def _retry_get(url: str, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    """HTTP GET with manual retry/backoff (in addition to HTTPAdapter retries).
       Returns JSON dict; if endpoint returns a list, normalize to {"data": list}.
    """
    sess = session or _build_session()
    for i in range(MAX_RETRIES):
        try:
            r = sess.get(url, headers=NSE_HEADERS, timeout=TIMEOUT, allow_redirects=True)
            if r.status_code == 200:
                try:
                    js = r.json()
                    return js if isinstance(js, dict) else {"data": js}
                except Exception:
                    pass
            if r.status_code in STATUS_FORCELIST:
                time.sleep(RETRY_BACKOFF ** i)
                continue
            break
        except Exception:
            time.sleep(RETRY_BACKOFF ** i)
            continue
    return {}

# -----------------------------
# Warm-up (cookie priming)
# -----------------------------
def nse_warmup(session: requests.Session) -> None:
    """Hit homepage + a cheap API to set cookies so subsequent API calls work."""
    try:
        session.get(NSE_BASE + "/", headers=NSE_HEADERS, timeout=TIMEOUT, allow_redirects=True)
        _jitter_sleep(0.2, 0.3)
        session.get(URL_ALL_INDICES, headers=NSE_HEADERS, timeout=TIMEOUT, allow_redirects=True)
    except Exception:
        pass

# -----------------------------
# Expiry handling
# -----------------------------
def _nearest_expiry(expiry_dates: List[str]) -> Optional[str]:
    """Pick the nearest expiry that is today or in the future."""
    if not expiry_dates:
        return None
    today = dt.date.today()

    def _parse(d: str) -> Optional[dt.date]:
        for fmt in ("%d-%b-%Y", "%d %b %Y", "%d-%B-%Y", "%d %B %Y", "%Y-%m-%d"):
            try:
                return dt.datetime.strptime(d.strip(), fmt).date()
            except Exception:
                continue
        return None

    parsed = []
    for raw in expiry_dates:
        d = _parse(str(raw))
        if d is not None:
            parsed.append((d, raw))
    if not parsed:
        return None

    parsed.sort(key=lambda t: t[0])
    for d, raw in parsed:
        if d >= today:
            return raw
    return parsed[-1][1]  # all past — return closest anyway

# -----------------------------
# Fetchers
# -----------------------------
def fetch_all_indices(session: Optional[requests.Session] = None) -> Dict[str, Any]:
    """Return JSON payload of all indices from NSE."""
    return _retry_get(URL_ALL_INDICES, session)


def fetch_fiidii(session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Fetch FII/DII flows; return tidy DataFrame with ISO date."""
    js = _retry_get(URL_FIIDII, session)
    data = js.get("data") or js.get("records") or js.get("fiiDii") or []
    if not data:
        return pd.DataFrame(columns=["category", "date", "buyValue", "sellValue", "netValue"])

    df = pd.DataFrame.from_records(data)
    rename = {
        "category": "category", "Category": "category",
        "date": "date", "Date": "date", "tradeDate": "date",
        "buyValue": "buyValue", "BuyValue": "buyValue", "buy": "buyValue",
        "sellValue": "sellValue", "SellValue": "sellValue", "sell": "sellValue",
        "netValue": "netValue", "NetValue": "netValue", "net": "netValue",
    }
    df = df.rename(columns=rename)
    for c in ("buyValue", "sellValue", "netValue"):
        if c in df.columns:
            df[c] = df[c].map(_to_float)
    if "date" in df.columns:
        df["date"] = df["date"].map(_normalize_date)

    keep = [c for c in ["category", "date", "buyValue", "sellValue", "netValue"] if c in df.columns]
    return df[keep] if keep else pd.DataFrame(columns=["category","date","buyValue","sellValue","netValue"])


def fetch_index_row(indices_json: Dict[str, Any], want_name: str) -> Optional[Dict[str, Any]]:
    """Find an index row by tolerant name match (trims/collapses whitespace, case-insensitive)."""
    want = _norm_name(want_name)
    for row in (indices_json.get("data", []) or []):
        idx_name = _norm_name(row.get("index", ""))
        if idx_name == want or want in idx_name or idx_name in want:
            return row
    return None


def fetch_india_vix(session: Optional[requests.Session] = None) -> float:
    """Return India VIX value. Primary: allIndices → last/ltp; Fallback: INDIAVIX OC underlyingValue."""
    idx = fetch_all_indices(session)
    row = fetch_index_row(idx, "INDIA VIX")
    if row:
        for key in ("last", "ltp", "lastPrice", "lastTradedPrice", "value", "changeClose"):
            v = _to_float(row.get(key))
            if v > 0:
                return v
    oc = _retry_get(URL_OPTION_CHAIN.format(symbol="INDIAVIX"), session)
    uv = ((oc or {}).get("records", {}) or {}).get("underlyingValue")
    try:
        return float(uv) if uv is not None else float("nan")
    except Exception:
        return float("nan")


def fetch_nifty_spot(session: Optional[requests.Session] = None) -> Optional[float]:
    """Return NIFTY spot price; primary: OC underlyingValue; fallback: allIndices last/ltp."""
    oc = _retry_get(URL_OPTION_CHAIN.format(symbol="NIFTY"), session)
    uv = ((oc or {}).get("records", {}) or {}).get("underlyingValue")
    if uv is not None:
        try:
            v = float(uv)
            if math.isfinite(v) and v > 0:
                return v
        except Exception:
            pass

    idx = fetch_all_indices(session)
    row = fetch_index_row(idx, "NIFTY 50") or fetch_index_row(idx, "NIFTY50")
    if row:
        for key in ("last", "ltp", "lastPrice", "lastValue", "value"):
            v = _to_float(row.get(key))
            if v > 0:
                return v
    return None


def fetch_nifty_option_chain_nearest_expiry(session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Return tidy CE/PE dataframe for the *nearest* expiry only, with robust fallbacks."""
    oc = _retry_get(URL_OPTION_CHAIN.format(symbol="NIFTY"), session)
    rec = (oc or {}).get("records", {}) or {}
    rows = rec.get("data", []) or []
    expiries = [str(x) for x in (rec.get("expiryDates", []) or [])]

    def _aggregate(_rows: List[dict]) -> pd.DataFrame:
        agg: Dict[float, Dict[str, float]] = {}
        for r in _rows:
            k = _to_float(r.get("strikePrice"))
            if k <= 0:
                continue
            ce = r.get("CE", {}) or {}
            pe = r.get("PE", {}) or {}
            e = agg.setdefault(k, {
                "strike": k,
                "ce_oi": 0.0, "ce_chg_oi": 0.0, "ce_vol": 0.0, "ce_iv": 0.0,
                "pe_oi": 0.0, "pe_chg_oi": 0.0, "pe_vol": 0.0, "pe_iv": 0.0,
            })
            e["ce_oi"]     += _to_float(ce.get("openInterest"))
            e["ce_chg_oi"] += _to_float(ce.get("changeinOpenInterest"))
            e["ce_vol"]    += _to_float(ce.get("totalTradedVolume"))
            iv = _to_float(ce.get("impliedVolatility"))
            if iv > 0: e["ce_iv"] = iv

            e["pe_oi"]     += _to_float(pe.get("openInterest"))
            e["pe_chg_oi"] += _to_float(pe.get("changeinOpenInterest"))
            e["pe_vol"]    += _to_float(pe.get("totalTradedVolume"))
            iv = _to_float(pe.get("impliedVolatility"))
            if iv > 0: e["pe_iv"] = iv

        cols = ["strike","ce_oi","ce_chg_oi","ce_vol","ce_iv","pe_oi","pe_chg_oi","pe_vol","pe_iv"]
        if not agg:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(list(agg.values()), columns=cols).sort_values("strike").reset_index(drop=True)

    # Try nearest expiry
    nearest = _nearest_expiry(expiries) if expiries else None
    filt = [r for r in rows if str(r.get("expiryDate","")).strip() == nearest] if nearest else []
    df = _aggregate(filt)
    if not df.empty:
        return df

    # Fallback: pick expiry with most rows
    if rows:
        buckets: Dict[str, List[dict]] = {}
        for r in rows:
            buckets.setdefault(str(r.get("expiryDate","")).strip(), []).append(r)
        if buckets:
            best_expiry = max(buckets.items(), key=lambda kv: len(kv[1]))[0]
            df = _aggregate(buckets[best_expiry])
            if not df.empty:
                return df

    # Final fallback: aggregate across all rows
    return _aggregate(rows)

# -----------------------------
# Metrics & labels
# -----------------------------
def compute_pcr_near_atm(chain: pd.DataFrame, spot: float, pct_window: float = 0.03) -> float:
    """Put/Call OI Ratio using nearest-expiry chain within ±pct_window of spot."""
    if chain is None or chain.empty or not math.isfinite(spot) or spot <= 0:
        return 1.0
    lower, upper = spot * (1 - pct_window), spot * (1 + pct_window)
    sub = chain[(chain["strike"] >= lower) & (chain["strike"] <= upper)]
    if sub.empty:
        sub = chain
    ce_oi = float(sub["ce_oi"].sum())
    pe_oi = float(sub["pe_oi"].sum())
    return (pe_oi / ce_oi) if ce_oi > 0 else 1.0


def market_sentiment_from_pcr(pcr: float) -> str:
    if not math.isfinite(pcr):
        return "Neutral"
    if pcr < 0.8:
        return "Bearish"
    if pcr > 1.2:
        return "Bullish"
    return "Neutral"


def market_regime_from_vix(vix: float, baseline: float = 12.0) -> str:
    if not math.isfinite(vix) or vix <= 0:
        return "Unknown"
    if vix < max(10.0, baseline * 0.9):
        return "Calm"
    if vix > max(16.0, baseline * 1.2):
        return "Stressed"
    return "Normal"

# -----------------------------
# Attempt + healing passes
# -----------------------------
def _attempt_snapshot(sess: requests.Session) -> dict:
    """One pass: warm-up and fetch everything once."""
    nse_warmup(sess); _jitter_sleep()

    fii = fetch_fiidii(sess); _jitter_sleep()
    oc  = fetch_nifty_option_chain_nearest_expiry(sess); _jitter_sleep()
    spot = fetch_nifty_spot(sess) or float("nan"); _jitter_sleep()
    vix  = fetch_india_vix(sess); _jitter_sleep()

    pcr = compute_pcr_near_atm(oc, spot, 0.03)
    sentiment = market_sentiment_from_pcr(pcr)
    regime = market_regime_from_vix(vix)

    return {
        "fii": fii, "oc": oc, "spot": spot, "vix": vix,
        "pcr": pcr, "sentiment": sentiment, "regime": regime,
        "ok_fii": True,                 # CSV gets written even if empty
        "ok_oc": not oc.empty,
        "ok_spot": math.isfinite(spot),
        "ok_vix": math.isfinite(vix),
    }


def _heal_missing(sess: requests.Session, snap: dict) -> dict:
    """Heal only missing pieces with alternate order & fresh warm-up."""
    # If OC missing → re-warm + OC again
    if not snap["ok_oc"]:
        nse_warmup(sess); _jitter_sleep()
        oc = fetch_nifty_option_chain_nearest_expiry(sess)
        if not oc.empty:
            snap["oc"] = oc
            snap["ok_oc"] = True
            if math.isfinite(snap["spot"]):
                snap["pcr"] = compute_pcr_near_atm(oc, snap["spot"], 0.03)
                snap["sentiment"] = market_sentiment_from_pcr(snap["pcr"])

    # If Spot missing → indices-first path explicitly
    if not snap["ok_spot"]:
        nse_warmup(sess); _jitter_sleep()
        idx = fetch_all_indices(sess)
        row = fetch_index_row(idx, "NIFTY 50") or fetch_index_row(idx, "NIFTY50")
        spot = None
        if row:
            for key in ("last","ltp","lastPrice","lastValue","value"):
                v = _to_float(row.get(key))
                if v > 0: spot = v; break
        if not spot:
            spot = fetch_nifty_spot(sess) or float("nan")
        snap["spot"] = spot
        snap["ok_spot"] = math.isfinite(spot)
        if snap["ok_spot"] and snap["ok_oc"]:
            snap["pcr"] = compute_pcr_near_atm(snap["oc"], snap["spot"], 0.03)
            snap["sentiment"] = market_sentiment_from_pcr(snap["pcr"])

    # If VIX missing → indices-first then OC fallback again
    if not snap["ok_vix"]:
        nse_warmup(sess); _jitter_sleep()
        vix = fetch_india_vix(sess)
        snap["vix"] = vix
        snap["ok_vix"] = math.isfinite(vix)
        snap["regime"] = market_regime_from_vix(vix)

    return snap

# -----------------------------
# Main orchestration
# -----------------------------
def main() -> None:
    # Global wall-clock timeout
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(RUN_TIMEOUT_SECS)

    sess = _build_session()

    attempts = 3
    snapshot = None
    last_snap = None
    for i in range(1, attempts + 1):
        snap = _attempt_snapshot(sess)
        last_snap = snap
        if snap["ok_oc"] and snap["ok_spot"] and snap["ok_vix"]:
            snapshot = snap
            break
        snap = _heal_missing(sess, snap)
        if snap["ok_oc"] and snap["ok_spot"] and snap["ok_vix"]:
            snapshot = snap
            break
        _jitter_sleep(0.5, 0.8)

    if snapshot is None:
        snapshot = last_snap  # best effort

    # Unpack for saving/printing
    fii_dii_df = snapshot["fii"]
    oc_df      = snapshot["oc"]
    nifty_spot = snapshot["spot"]
    india_vix  = snapshot["vix"]
    pcr        = snapshot["pcr"]
    sentiment  = snapshot["sentiment"]
    regime     = snapshot["regime"]

    # Save outputs (always write all four)
    fii_dii_csv = str(DATE_DIR / "fii_dii.csv")
    oc_csv = str(DATE_DIR / "nifty_option_chain_nearest_expiry.csv")
    snapshot_json = str(DATE_DIR / "snapshot.json")
    snapshot_txt = str(DATE_DIR / "snapshot.txt")

    fii_dii_df.to_csv(fii_dii_csv, index=False)
    oc_df.to_csv(oc_csv, index=False)

    snapshot_meta = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "nifty_spot": None if not math.isfinite(nifty_spot) else round(float(nifty_spot), 2),
        "india_vix": None if not math.isfinite(india_vix) else round(float(india_vix), 2),
        "pcr_near_atm": round(float(pcr), 3) if math.isfinite(pcr) else None,
        "market_sentiment": sentiment,
        "market_regime": regime,
        "files": {"fii_dii_csv": fii_dii_csv, "option_chain_csv": oc_csv},
    }
    with open(snapshot_json, "w", encoding="utf-8") as f:
        json.dump(snapshot_meta, f, ensure_ascii=False, indent=2)

    # Human-friendly text snapshot
    lines: List[str] = []
    lines.append("FII/DII Data (head):")
    if fii_dii_df.empty:
        lines.append("  <no data>")
    else:
        lines.append(fii_dii_df.head(6).to_string(index=False))

    lines.append("")
    lines.append("NIFTY Option Chain — Nearest Expiry (head):")
    if oc_df.empty:
        lines.append("  <no data>")
    else:
        lines.append(oc_df.head(10).to_string(index=False))

    lines.append("")
    lines.append(f"NIFTY Spot: {nifty_spot if math.isfinite(nifty_spot) else 'nan'}")
    lines.append(f"India VIX: {india_vix if math.isfinite(india_vix) else 'nan'}")
    lines.append(f"PCR (±3% near ATM): {pcr:.3f}")
    lines.append(f"Market Sentiment: {sentiment}")
    lines.append(f"Market Regime: {regime}")

    text = "\n".join(lines)
    with open(snapshot_txt, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    # Display to console
    print(text)
    print("\nSaved:")
    print(f" - {fii_dii_csv}")
    print(f" - {oc_csv}")
    print(f" - {snapshot_json}")
    print(f" - {snapshot_txt}")

    # Disable global alarm
    signal.alarm(0)

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except TimeoutError as e:
        print(f"[FATAL] {e}")
        sys.exit(124)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)