"""
stochax_market/merge.py

Merge stochax-simulate predicted prices with realised actual prices from
V1.1.0 (upstox-historical) data, producing the unified CSV schema consumed
by visualize.plot_actual_vs_predicted.

Schema produced
---------------
step            int   — 0-indexed forecast step
predicted_price float — Monte Carlo mean prediction for that step (INR)
actual_price    float — realised Close from V1.1.0; NaN for forecast-only
                        steps (future dates) and trading holidays
date            str   — DD/MM/YY (matches the user's spreadsheet convention)

This module is library-only. The CLI lives in `stochax_market.cli:merge_app`
alongside the other Typer apps (simulate_app, fit_app, predict_app,
visualize_app, export_app), and is exposed as the `stochax-merge` entry
point in pyproject.toml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from stochax_market.data.loader import load_stock


_REQUIRED_INPUT_COLS = {"step", "predicted_price"}
_OUTPUT_SCHEMA       = ["step", "predicted_price", "actual_price", "date"]


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_start_date(
    start_date: Union[str, pd.Timestamp, None],
    df: pd.DataFrame,
) -> pd.Timestamp:
    """
    Determine the date corresponding to step 0 of the forecast.

    If `start_date` is None, default to the next business day after the
    latest date in the loaded historical DataFrame — the same convention
    used by predict.py when generating its forecast.

    Strings are parsed with `dayfirst=True` so the user can pass either the
    ISO 2026-04-18 form or the DD/MM/YY form that matches the on-disk CSV.
    """
    if start_date is None:
        last_known = df["Date"].max()
        return (last_known + pd.tseries.offsets.BDay(1)).normalize()

    if isinstance(start_date, pd.Timestamp):
        return start_date.normalize()

    return pd.to_datetime(start_date, dayfirst=True).normalize()


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def merge_predicted_with_actuals(
    sim_csv: Path,
    symbol: str,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    out: Optional[Path] = None,
) -> Path:
    """
    Augment a stochax-simulate output CSV with realised actual prices.

    The simulate CSV is expected to contain at least `step` and
    `predicted_price`. This function looks up the corresponding actual
    Close prices from V1.1.0 historical data (via `load_stock`) on a
    business-day grid anchored at `start_date`, and writes a CSV in the
    schema consumed by `visualize.plot_actual_vs_predicted`.

    Re-running this function in place is safe and idempotent: each call
    rebuilds the `actual_price` and `date` columns from scratch, so as
    additional days are realised the output picks them up automatically
    without leaving stale NaNs.

    Parameters
    ----------
    sim_csv    : Path to a stochax-simulate output CSV. Must contain
                 columns `step` and `predicted_price`. Any other columns
                 (including pre-existing `actual_price` / `date` from a
                 prior merge) are dropped and rebuilt.
    symbol     : Stock ticker. Used to load the V1.1.0 historical CSV via
                 `stochax_market.data.loader.load_stock`.
    start_date : Date corresponding to forecast step 0. Accepts a
                 `pd.Timestamp`, an ISO `YYYY-MM-DD` string, or the
                 DD/MM/YY string used in the on-disk sim CSV. If None,
                 defaults to the next business day after the latest date
                 in the loaded historical data — i.e. "predict from where
                 history ends", matching predict.py's convention.
    out        : Output path. If None, the input `sim_csv` is overwritten
                 in place (safe — the function is idempotent).

    Returns
    -------
    Path to the written CSV.

    Raises
    ------
    ValueError
        If `sim_csv` is missing required columns, or if the V1.1.0 loader
        returns a DataFrame without the expected `Date` / `Close` schema.
    """
    sim_csv = Path(sim_csv)
    sim     = pd.read_csv(sim_csv)

    missing = _REQUIRED_INPUT_COLS - set(sim.columns)
    if missing:
        raise ValueError(
            f"merge_predicted_with_actuals: {sim_csv} is missing columns {missing}. "
            f"Expected at least: {sorted(_REQUIRED_INPUT_COLS)}."
        )

    sim = sim.sort_values("step").reset_index(drop=True)
    n   = len(sim)

    # Load V1.1.0 historical data through stochax_market's own loader so the
    # column conventions (capitalised Date / Close) match what visualize.py
    # already relies on.
    df = load_stock(symbol)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(
            f"load_stock({symbol!r}) returned a DataFrame without Date/Close "
            "columns. Expected V1.1.0 schema with capitalised columns."
        )

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    anchor     = _resolve_start_date(start_date, df)

    # Forecast date grid: business days starting from `anchor` for n steps.
    forecast_dates = pd.bdate_range(anchor, periods=n)

    # Index actuals by date for O(1) lookup.
    actuals_by_date = (
        df.drop_duplicates(subset=["Date"])
          .set_index("Date")["Close"]
          .astype(float)
    )

    # Where a forecast date is in the future or falls on a holiday not in
    # the V1.1.0 data, .map() returns NaN — exactly what we want.
    actual_series = forecast_dates.to_series().map(actuals_by_date)

    merged = pd.DataFrame({
        "step":            sim["step"].astype(int).values,
        "predicted_price": sim["predicted_price"].astype(float).values,
        "actual_price":    actual_series.values,
        "date":            forecast_dates.strftime("%d/%m/%y"),
    })

    out_path = Path(out) if out is not None else sim_csv
    merged.to_csv(out_path, index=False)
    return out_path
