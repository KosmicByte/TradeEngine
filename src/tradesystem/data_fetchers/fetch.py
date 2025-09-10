"""
NSE data fetchers using `nsepython` only.
- FII/DII flows
- Market sentiment (breadth + PCR + VIX Δ% + NIFTY return)
- Option chain (nearest expiry, tidy columns)
- India VIX (with previous close safety)
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from nsepython import nse_fiidii, nse_optionchain_scrapper, nsefetch


# ------------------------------- Retry Wrapper -------------------------------

def _retry(fn, *args, retries: int = 3, delay: float = 0.5, **kwargs):
    """
    Execute `fn` with simple exponential backoff. Raises last exception on failure.
    """
    backoff = delay
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 2


# ----------------------------- Core NSE Utilities ----------------------------

def _fetch_all_indices() -> List[Dict[str, Any]]:
    """
    Pulls the 'allIndices' feed and returns a list of index rows.
    Handles payload variants: { "data": [...] } or [...].

    Row keys commonly observed (not guaranteed):
      - index / indexName (str)
      - last / lastPrice / lastValue / value / ltp (float)
      - previousClose / prevClose (float)
      - variation / percentChange (floats)
      - advances / declines (ints) [not always available]
    """
    data = _retry(nsefetch, "https://www.nseindia.com/api/allIndices")
    items = data.get("data", data)
    if not isinstance(items, list):
        raise ValueError("Unexpected allIndices payload shape")
    return items


def _norm_index_name(row: Dict[str, Any]) -> str:
    return (row.get("index") or row.get("indexName") or "").strip().upper()


def _get_float(row: Dict[str, Any], keys: Iterable[str], default: Optional[float] = None) -> Optional[float]:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            try:
                return float(row[k])
            except Exception:
                pass
    return default


def _get_index_row(name: str, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Returns the first matching index row (case-insensitive, tolerant to spacing).
    """
    name_u = name.strip().upper()
    for r in rows:
        if _norm_index_name(r) == name_u:
            return r
    return None


# ------------------------------- Public Fetchers -----------------------------

def fetch_fii_dii() -> pd.DataFrame:
    """
    Returns latest FII/DII activity as a DataFrame with columns:
    [category, date, buyValue, sellValue, netValue]
    """
    df = _retry(nse_fiidii)
    return pd.DataFrame(df)[["category", "date", "buyValue", "sellValue", "netValue"]]


def fetch_nifty_vix() -> float:
    """
    Returns latest India VIX value from allIndices. Falls back sanely across key variants.
    """
    rows = _fetch_all_indices()
    vix = _get_index_row("INDIA VIX", rows) or _get_index_row("INDIAVIX", rows)
    if not vix:
        raise KeyError("India VIX not present in allIndices payload")

    last = _get_float(vix, ("last", "lastPrice", "lastValue", "value", "ltp"))
    if last is None:
        raise KeyError("India VIX found but no usable price field")
    return float(last)


def _fetch_vix_change_pct() -> float:
    """
    Returns % change for India VIX if previous close is available; else 0.0.
    """
    rows = _fetch_all_indices()
    vix = _get_index_row("INDIA VIX", rows) or _get_index_row("INDIAVIX", rows)
    if not vix:
        return 0.0

    last = _get_float(vix, ("last", "lastPrice", "lastValue", "value", "ltp"), default=None)
    prev = _get_float(vix, ("previousClose", "prevClose"), default=None)
    if last is None or prev in (None, 0.0):
        return 0.0
    return (last - prev) / prev * 100.0


def _fetch_nifty_return_pct() -> float:
    """
    Returns NIFTY 50 return in % (last vs previousClose) if available; else 0.0.
    """
    rows = _fetch_all_indices()
    # Accept common variants for NIFTY 50 name on this feed
    for name in ("NIFTY 50", "NIFTY50", "NIFTY 50 INDEX"):
        row = _get_index_row(name, rows)
        if row:
            last = _get_float(row, ("last", "lastPrice", "lastValue", "value", "ltp"))
            prev = _get_float(row, ("previousClose", "prevClose"))
            if last is not None and prev not in (None, 0.0):
                return (last - prev) / prev * 100.0
            break
    return 0.0


def _fetch_market_breadth_ratio() -> float:
    """
    Computes a market breadth ratio using the allIndices feed.

    Strategy:
    1) If any row exposes `advances` and `declines` (rare but possible), compute sum(adv)/sum(dec).
    2) Else, proxy breadth by counting indices with positive vs negative % change across a
       curated basket (exclude VIX). If pos==neg==0, return 1.0.

    Returns:
        breadth_ratio = positives / max(1, negatives); if negatives=0 but positives>0, cap at 2.0.
    """
    rows = _fetch_all_indices()

    # Try explicit advances/declines if present on any rows
    adv_total = 0
    dec_total = 0
    for r in rows:
        adv = r.get("advances")
        dec = r.get("declines")
        if isinstance(adv, (int, float)) and isinstance(dec, (int, float)):
            adv_total += int(adv)
            dec_total += int(dec)
    if (adv_total + dec_total) > 0:
        return (adv_total / max(1, dec_total)) if dec_total > 0 else 2.0

    # Fallback: sign of percentage change across a reasonable basket (exclude VIX)
    basket_names = {
        "NIFTY 50", "NIFTY BANK", "NIFTY FIN SERVICE", "NIFTY NEXT 50",
        "NIFTY MIDCAP 100", "NIFTY SMALLCAP 100", "NIFTY IT", "NIFTY FMCG",
        "NIFTY PHARMA", "NIFTY AUTO", "NIFTY METAL"
    }
    pos = neg = 0
    for r in rows:
        name = _norm_index_name(r)
        if name in ("INDIA VIX", "INDIAVIX"):
            continue
        if basket_names and name not in basket_names:
            continue
        pct = _get_float(r, ("percentChange", "variation"))  # percentChange usually present
        if pct is None:
            # derive if last/prevClose available
            last = _get_float(r, ("last", "lastPrice", "lastValue", "value", "ltp"))
            prev = _get_float(r, ("previousClose", "prevClose"))
            if last is not None and prev not in (None, 0.0):
                pct = (last - prev) / prev * 100.0
        if pct is None:
            continue
        if pct > 0:
            pos += 1
        elif pct < 0:
            neg += 1

    if pos == 0 and neg == 0:
        return 1.0
    if neg == 0:
        return 2.0  # cap to avoid infinity
    return pos / neg


def _nearest_expiry(records: Dict[str, Any]) -> Optional[str]:
    """
    Return the first expiry from records['expiryDates'] (NSE orders by nearest).
    """
    exps = records.get("expiryDates") or []
    return exps[0] if exps else None


def _underlying_value(records: Dict[str, Any]) -> Optional[float]:
    return _get_float(records, ("underlyingValue",), default=None)


def _select_near_atm_strikes(data: List[Dict[str, Any]], spot: Optional[float], width: int = 10) -> List[Dict[str, Any]]:
    """
    Return ~2*width rows around ATM (by absolute |strike-spot|). If spot is None, return all.
    """
    if spot is None:
        return data
    scored = []
    for d in data:
        k = d.get("strikePrice")
        if k is None:
            continue
        scored.append((abs(k - spot), d))
    scored.sort(key=lambda x: x[0])
    return [d for _, d in scored[: 2 * width]]


# ----------------------------- Option Chain Fetch ----------------------------

def fetch_nse_option_chain(symbol: str = "NIFTY") -> pd.DataFrame:
    """
    Returns a tidy option chain (nearest expiry) for the given index symbol.
    Columns:
      strike, ce_oi, ce_chg_oi, ce_vol, ce_iv, pe_oi, pe_chg_oi, pe_vol, pe_iv
    """
    payload = _retry(nse_optionchain_scrapper, symbol)
    records = payload.get("records", {})
    data: List[Dict[str, Any]] = payload.get("records", {}).get("data", [])
    if not isinstance(data, list):
        return pd.DataFrame(columns=["strike", "ce_oi", "ce_chg_oi", "ce_vol", "ce_iv",
                                     "pe_oi", "pe_chg_oi", "pe_vol", "pe_iv"])

    # Filter to nearest expiry to avoid duplicate strikes across expiries
    target_exp = _nearest_expiry(records)
    rows = []
    for item in data:
        k = item.get("strikePrice")
        ce = item.get("CE") or {}
        pe = item.get("PE") or {}

        # Keep only rows where CE/PE match nearest expiry (if available)
        ce_exp_ok = (not ce) or (ce.get("expiryDate") == target_exp)
        pe_exp_ok = (not pe) or (pe.get("expiryDate") == target_exp)
        if k is None or not (ce_exp_ok or pe_exp_ok):
            continue

        rows.append({
            "strike": k,
            "ce_oi": ce.get("openInterest", 0) or 0,
            "ce_chg_oi": ce.get("changeinOpenInterest", 0) or 0,
            "ce_vol": ce.get("totalTradedVolume", 0) or 0,
            "ce_iv": ce.get("impliedVolatility", 0) or 0.0,
            "pe_oi": pe.get("openInterest", 0) or 0,
            "pe_chg_oi": pe.get("changeinOpenInterest", 0) or 0,
            "pe_vol": pe.get("totalTradedVolume", 0) or 0,
            "pe_iv": pe.get("impliedVolatility", 0) or 0.0,
        })

    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    return df


# ----------------------------- PCR and Sentiment -----------------------------

def _compute_pcr_near_atm(symbol: str = "NIFTY", width: int = 10) -> float:
    """
    Computes Put/Call OI Ratio (PCR) using only strikes near ATM for the
    nearest expiry. This avoids noisy far OTM strikes across multiple expiries.
    """
    payload = _retry(nse_optionchain_scrapper, symbol)
    records = payload.get("records", {})
    data = records.get("data", [])
    if not data:
        return 1.0

    target_exp = _nearest_expiry(records)
    spot = _underlying_value(records)

    # Keep nearest-expiry rows only
    exp_rows = []
    for item in data:
        ce = item.get("CE") or {}
        pe = item.get("PE") or {}
        if (ce and ce.get("expiryDate") != target_exp) and (pe and pe.get("expiryDate") != target_exp):
            continue
        exp_rows.append(item)

    # Slice around ATM
    near = _select_near_atm_strikes(exp_rows, spot, width=width)

    ce_oi = sum((x.get("CE") or {}).get("openInterest", 0) or 0 for x in near)
    pe_oi = sum((x.get("PE") or {}).get("openInterest", 0) or 0 for x in near)
    if ce_oi <= 0:
        return 1.0
    return float(pe_oi) / float(ce_oi)


def fetch_market_sentiment() -> str:
    """
    Composes a simple market sentiment signal from four components:
      1) Breadth ratio from allIndices (positives/negatives or advances/declines).
      2) PCR near ATM on NIFTY (nearest expiry).
      3) India VIX % change vs previous close.
      4) NIFTY 50 return % vs previous close.

    Scoring:
      - Breadth > 1.1 -> +1; < 0.9 -> -1; else 0
      - PCR in [0.9, 1.1] -> 0; > 1.1 -> -1; < 0.9 -> +1
      - VIX Δ% > +5 -> -1; < -5 -> +1; else 0
      - NIFTY return > +0.5 -> +1; < -0.5 -> -1; else 0
    """
    breadth = _fetch_market_breadth_ratio()
    pcr = _compute_pcr_near_atm("NIFTY", width=10)
    vix_delta = _fetch_vix_change_pct()
    nifty_ret = _fetch_nifty_return_pct()

    score = 0
    score += 1 if breadth > 1.1 else (-1 if breadth < 0.9 else 0)
    score += 0 if 0.9 <= pcr <= 1.1 else (-1 if pcr > 1.1 else 1)
    score += -1 if vix_delta > 5.0 else (1 if vix_delta < -5.0 else 0)
    score += 1 if nifty_ret > 0.5 else (-1 if nifty_ret < -0.5 else 0)

    return "Positive" if score > 1 else ("Negative" if score < -1 else "Neutral")


# ------------------------------ ATR Regime Filter ----------------------------

def atr_trend_filter(df_ohlc: pd.DataFrame, window: int = 14) -> Tuple[str, pd.DataFrame]:
    """
    Computes a simple ATR slope on an OHLC DataFrame and classifies regime.

    Args:
        df_ohlc: DataFrame with columns ['high','low','close'] (at minimum).
        window: ATR window length (default 14).

    Returns:
        (regime, tail_df) where regime is "Volatile" if recent ATR slope > 0, else "Calm",
        and tail_df shows the last ATR/ATR_slope rows for inspection.
    """
    df = df_ohlc.copy()
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("df_ohlc must contain 'high','low','close' columns")

    df["H-L"] = df["high"] - df["low"]
    df["H-PC"] = (df["high"] - df["close"].shift(1)).abs()
    df["L-PC"] = (df["low"] - df["close"].shift(1)).abs()
    tr = df[["H-L", "H-PC", "L-PC"]].max(axis=1)

    df["atr"] = tr.rolling(window=window, min_periods=window).mean()
    df["atr_slope"] = df["atr"].diff()

    # Slope over the most recent 5 values (where available)
    slope = df["atr_slope"].tail(5).mean()
    regime = "Volatile" if (pd.notna(slope) and slope > 0) else "Calm"
    return regime, df[["atr", "atr_slope"]].tail(5)


# ---------------------------------- Script -----------------------------------

if __name__ == "__main__":
    print("FII/DII Data:")
    print(fetch_fii_dii())

    print("\nMarket Sentiment:")
    print(fetch_market_sentiment())

    print("\nNIFTY Option Chain (nearest expiry, first 5 rows):")
    print(fetch_nse_option_chain("NIFTY").head())

    print("\nIndia VIX:")
    print(fetch_nifty_vix())