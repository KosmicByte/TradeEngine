from __future__ import annotations

import math
import time
from typing import Callable, Any, Tuple, Optional

import pandas as pd
from nsepython import (
    nse_fiidii,
    nse_optionchain_scrapper,
    nsefetch,                  # allowed as per spec for official NSE endpoints
)

# ---------------------------
# Utilities
# ---------------------------

def _retry(fn: Callable[..., Any], *args, retries: int = 3, **kwargs) -> Any:
    """Generic retry wrapper with 0.5x, 1x, 2x backoff."""
    delay = 0.5
    last_exc: Optional[Exception] = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if i == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    # should never get here
    raise last_exc or RuntimeError("Unknown retry failure")


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if isinstance(x, str):
            x = x.replace(",", "").replace(" ", "").replace("\u2212", "-").strip()
        return float(x)
    except Exception:
        return default


def _normalize_date(s: Any) -> Any:
    """Return YYYY-MM-DD if s looks like a date; else return original."""
    try:
        return pd.to_datetime(s).date().isoformat()
    except Exception:
        return s


# ---------------------------
# 1) FII / DII
# ---------------------------

def fetch_fii_dii() -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      category, date, buyValue, sellValue, netValue
    Data source: nsepython.nse_fiidii()
    """
    raw = _retry(nse_fiidii)
    df = pd.DataFrame(raw)
    # Accept common variants in key names and normalize
    rename_map = {
        "category": "category",
        "Category": "category",
        "fiiDii": "category",
        "date": "date",
        "Date": "date",
        "buyValue": "buyValue",
        "BuyValue": "buyValue",
        "buy_value": "buyValue",
        "sellValue": "sellValue",
        "SellValue": "sellValue",
        "sell_value": "sellValue",
        "netValue": "netValue",
        "NetValue": "netValue",
        "net_value": "netValue",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    cols = [c for c in ["category", "date", "buyValue", "sellValue", "netValue"] if c in df.columns]
    df = df[cols].copy()

    # Types + date normalization
    if "date" in df.columns:
        df["date"] = df["date"].map(_normalize_date)
    for c in ("buyValue", "sellValue", "netValue"):
        if c in df.columns:
            df[c] = df[c].map(_to_float)

    # Keep only the typical FII/FPI + DII rows if present
    if "category" in df.columns:
        mask = df["category"].astype(str).str.upper().isin(["FII", "FPI", "DII"])
        if mask.any():
            df = df[mask].reset_index(drop=True)

    return df


# ---------------------------
# 3) Option Chain (indices)
# ---------------------------

def fetch_nse_option_chain(symbol: str = "NIFTY") -> pd.DataFrame:
    """
    Returns tidy chain with:
      strike, ce_oi, ce_chg_oi, ce_vol, ce_iv, pe_oi, pe_chg_oi, pe_vol, pe_iv
    Data source: nsepython.nse_optionchain_scrapper(symbol)
    """
    payload = _retry(nse_optionchain_scrapper, symbol)
    records = (payload or {}).get("records", {})
    data = records.get("data", []) or []

    rows = []
    for item in data:
        strike = item.get("strikePrice")
        if strike is None:
            continue
        ce = item.get("CE", {}) or {}
        pe = item.get("PE", {}) or {}
        rows.append({
            "strike": strike,
            "ce_oi": ce.get("openInterest", 0),
            "ce_chg_oi": ce.get("changeinOpenInterest", 0),
            "ce_vol": ce.get("totalTradedVolume", 0),
            "ce_iv": ce.get("impliedVolatility", 0.0),
            "pe_oi": pe.get("openInterest", 0),
            "pe_chg_oi": pe.get("changeinOpenInterest", 0),
            "pe_vol": pe.get("totalTradedVolume", 0),
            "pe_iv": pe.get("impliedVolatility", 0.0),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Clean numeric types
    num_cols = [c for c in df.columns if c != "strike"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df = df.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)
    return df


# ---------------------------
# 4) VIX + Volatility Regime
# ---------------------------

def fetch_nifty_vix() -> float:
    # Primary: allIndices (reliable for VIX)
    try:
        js = _retry(nsefetch, "https://www.nseindia.com/api/allIndices")
        items = (js or {}).get("data", []) or []
        row = next((x for x in items if str(x.get("index", "")).upper() in ("INDIA VIX","INDIAVIX")), None)
        if row:
            val = _to_float(row.get("last")) or _to_float(row.get("lastPrice")) \
                  or _to_float(row.get("lastValue")) or _to_float(row.get("value")) \
                  or _to_float(row.get("ltp"))
            if val:
                return float(val)
    except Exception:
        pass

    # Fallback: OC (often missing for INDIAVIX)
    try:
        vix_payload = _retry(nse_optionchain_scrapper, "INDIAVIX")
        uv = (vix_payload or {}).get("records", {}).get("underlyingValue", None)
        return float(uv) if uv is not None else float("nan")
    except Exception:
        return float("nan")

def atr_trend_filter(df_ohlc: pd.DataFrame, window: int = 14) -> Tuple[str, pd.DataFrame]:
    """
    Classic ATR slope regime. If you have OHLC for NIFTY, pass it here.
    Returns (regime, tail_df_with_atr_and_slope).
    """
    if df_ohlc is None or df_ohlc.empty:
        return "Calm", pd.DataFrame()

    df = df_ohlc.copy()
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError("df_ohlc must contain columns: high, low, close")

    df["H-L"] = df["high"] - df["low"]
    df["H-PC"] = (df["high"] - df["close"].shift(1)).abs()
    df["L-PC"] = (df["low"] - df["close"].shift(1)).abs()
    tr = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["atr"] = tr.rolling(window=window, min_periods=window).mean()
    df["atr_slope"] = df["atr"].diff()

    slope = df["atr_slope"].tail(5).mean()
    regime = "Volatile" if slope > 0 else "Calm"
    return regime, df[["atr", "atr_slope"]].tail(1)


def _vix_regime_fallback() -> str:
    """
    If OHLC is not provided: classify regime using VIX % delta if previous close available
    via NSE indices API (through nsefetch). If not obtainable, return 'Calm'.
    """
    try:
        # Pull index quote for INDIAVIX via official endpoint with nsefetch (allowed by spec)
        js = _retry(nsefetch, "https://www.nseindia.com/api/allIndices")
        # Find the INDIAVIX line
        items = (js or {}).get("data", []) or []
        row = next((x for x in items if str(x.get("index", "")).upper() == "INDIA VIX"), None)
        if not row:
            return "Calm"
        last = _to_float(row.get("last"))
        prev = _to_float(row.get("previousClose"))
        if last and prev:
            delta_pct = (last - prev) * 100.0 / prev
            if delta_pct > 5:
                return "Volatile"
            if delta_pct < -5:
                return "Calm"
        return "Calm"
    except Exception:
        return "Calm"


# ---------------------------
# 2) Sentiment (no web-scrape)
# ---------------------------

def _market_breadth_ratio() -> float:
    """
    Try to compute breadth (advancers/decliners) from NSE index universe
    using official endpoints via nsefetch. If not obtainable, return 1.0.
    """
    try:
        # Use allIndices and aggregate signs by constituent change if available,
        # else approximate via headline advances/declines if present.
        js = _retry(nsefetch, "https://www.nseindia.com/api/allIndices")
        data = (js or {}).get("data", []) or []
        adv = 0
        dec = 0
        for row in data:
            # many rows (broad indices). Use net change sign as a coarse proxy breadth.
            chg = _to_float(row.get("variation"))  # % variation
            if chg > 0:
                adv += 1
            elif chg < 0:
                dec += 1
        if adv == 0 and dec == 0:
            return 1.0
        return (adv / max(dec, 1.0))
    except Exception:
        return 1.0


def _nifty_return_pct() -> float:
    """
    NIFTY return % from option chain 'underlyingValue' vs 'prevClose' if discoverable.
    If prev close not present, fallback to 0.0 (neutral contribution).
    """
    try:
        oc = _retry(nse_optionchain_scrapper, "NIFTY")
        rec = (oc or {}).get("records", {}) or {}
        last = _to_float(rec.get("underlyingValue"))
        # try to obtain prev close via indices API
        q = _retry(nsefetch, "https://www.nseindia.com/api/allIndices")
        data = (q or {}).get("data", []) or []
        row = next((x for x in data if str(x.get("index", "")).upper() in ("NIFTY 50", "NIFTY50")), None)
        prev = _to_float(row.get("previousClose")) if row else 0.0
        if last and prev:
            return (last - prev) * 100.0 / prev
        return 0.0
    except Exception:
        return 0.0


def _pcr_from_chain_near_atm(symbol: str = "NIFTY") -> float:
    """
    Compute PCR from summed OI across strikes (simple proxy). If you want "near ATM",
    you can tighten the window around underlying.
    """
    oc = _retry(nse_optionchain_scrapper, symbol)
    rec = (oc or {}).get("records", {})
    data = rec.get("data", []) or []
    uv = _to_float(rec.get("underlyingValue"))

    # Choose strikes within +/- 3% of underlying to focus near ATM
    rows = []
    for item in data:
        k = _to_float(item.get("strikePrice"))
        if not (k and uv):
            continue
        if abs(k - uv) / uv <= 0.03:
            rows.append(item)

    if not rows:
        rows = data  # fallback to all

    ce_oi = sum(_to_float(x.get("CE", {}).get("openInterest", 0)) for x in rows)
    pe_oi = sum(_to_float(x.get("PE", {}).get("openInterest", 0)) for x in rows)
    if ce_oi == 0:
        return 1.0
    return pe_oi / ce_oi


def fetch_market_sentiment() -> str:
    """
    Replace site-scrape headlines with a small additive signal purely from NSE data:
      + Breadth (adv/dec ratio) from allIndices
      + PCR (NIFTY) from option chain
      + VIX Δ% sign (from allIndices)
      + NIFTY return % (from allIndices + OC)
    Scoring:
      Breadth > 1.1 => +1, < 0.9 => -1
      PCR 0.9–1.1 => 0; >1.1 => -1; <0.9 => +1
      VIX Δ% > +5% => -1; < -5% => +1
      NIFTY ret > +0.5% => +1; < -0.5% => -1
    """
    # Breadth
    breadth = _market_breadth_ratio()

    # PCR
    pcr = _pcr_from_chain_near_atm("NIFTY")

    # VIX delta %
    vix_state = _retry(nsefetch, "https://www.nseindia.com/api/allIndices")
    vix_data = (vix_state or {}).get("data", []) or []
    vix_row = next((x for x in vix_data if str(x.get("index", "")).upper() == "INDIA VIX"), None)
    if vix_row:
        vix_last = _to_float(vix_row.get("last"))
        vix_prev = _to_float(vix_row.get("previousClose"))
        vix_delta = (vix_last - vix_prev) * 100.0 / vix_prev if (vix_last and vix_prev) else 0.0
    else:
        # if not available, neutralize VIX contribution
        vix_delta = 0.0

    # NIFTY return %
    nifty_ret = _nifty_return_pct()

    score = 0
    score += (1 if breadth > 1.1 else (-1 if breadth < 0.9 else 0))
    score += (0 if 0.9 <= pcr <= 1.1 else (-1 if pcr > 1.1 else 1))
    score += (-1 if vix_delta > 5 else (1 if vix_delta < -5 else 0))
    score += (1 if nifty_ret > 0.5 else (-1 if nifty_ret < -0.5 else 0))

    return "Positive" if score > 1 else ("Negative" if score < -1 else "Neutral")


# ---------------------------
# Lightweight self-checks
# ---------------------------

def _self_test():
    # FII/DII
    fii_df = fetch_fii_dii()
    assert set(["category", "date", "buyValue", "sellValue", "netValue"]).issubset(fii_df.columns)
    # Option chain
    oc_df = fetch_nse_option_chain("NIFTY")
    if not oc_df.empty:
        expect = ["strike","ce_oi","ce_chg_oi","ce_vol","ce_iv","pe_oi","pe_chg_oi","pe_vol","pe_iv"]
        assert all(c in oc_df.columns for c in expect)
    # VIX
    v = fetch_nifty_vix()
    assert isinstance(v, float) or math.isnan(v)
    # Sentiment
    s = fetch_market_sentiment()
    assert s in {"Positive", "Neutral", "Negative"}


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    # 1) FII/DII
    fii = fetch_fii_dii()
    print("FII/DII sample:")
    print(fii.head(2).to_string(index=False))

    # 2) Sentiment
    print("\nMarket Sentiment:", fetch_market_sentiment())

    # 3) Option Chain
    chain = fetch_nse_option_chain("NIFTY")
    print("\nOption Chain (NIFTY) head:")
    print(chain.head().to_string(index=False))

    # 4) VIX + Regime (fallback without OHLC)
    vix_val = fetch_nifty_vix()
    regime = _vix_regime_fallback()
    print(f"\nIndia VIX: {vix_val}")
    print("Market Regime:", regime)

    # quick internal assertions
    try:
        _self_test()
        print("\n[Self-test] OK")
    except AssertionError as e:
        print("\n[Self-test] FAILED:", e)