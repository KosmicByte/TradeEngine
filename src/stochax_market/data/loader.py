"""
Loader for NIFTY50 stock data from local CSV files.

Files are expected at `./data/{SYMBOL}.csv` (relative to the current
working directory) with the V1.1.0 capitalised schema:

    Date, Symbol, Series, Prev Close, Open, High, Low, Close, Volume,
    VWAP, Turnover, Trades, Deliverable Volume, %Deliverble

Date column may be tz-aware (e.g. '2026-05-08 00:00:00+05:30' as written
by the V1.1.0 Upstox fetcher) or tz-naive ISO. The loader normalises both
forms to tz-naive midnight timestamps. This is essential for downstream
matching against the tz-naive business-day grids that `stochax-merge`,
`stochax-diagnose`, and `visualize.run_all` construct via `pd.bdate_range`
— without this normalisation, every date-keyed lookup silently returns
NaN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd


DEFAULT_DATA_DIR = Path("data")
REQUIRED_COLUMNS = {"Date", "Close"}


def load_stock(
    symbol: str,
    data_dir: Optional[Union[Path, str]] = None,
) -> pd.DataFrame:
    """
    Load a NIFTY50 stock's historical data from a local CSV.

    Parameters
    ----------
    symbol   : Stock ticker (e.g. 'RELIANCE'). Resolves to
               `{data_dir}/{symbol}.csv`.
    data_dir : Directory containing per-symbol CSVs. Defaults to
               `./data/` relative to the current working directory.

    Returns
    -------
    DataFrame sorted by Date ascending. The `Date` column is normalised
    to tz-naive midnight timestamps regardless of whether the source
    file had a timezone offset — the local calendar date as written is
    preserved (tz-aware IST timestamps are not shifted to UTC).

    Raises
    ------
    FileNotFoundError
        If `{data_dir}/{symbol}.csv` does not exist. The error includes
        the absolute path that was checked, so the remedy is obvious
        from the traceback.
    ValueError
        If the loaded DataFrame is missing required columns.
    """
    base = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    path = base / f"{symbol}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"load_stock({symbol!r}): expected file at {path.resolve()}.\n"
            f"Place a V1.1.0 fetcher output CSV there "
            f"(capitalised schema: Date, Close, Open, High, Low, VWAP, ...)."
        )

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing required columns {sorted(missing)}. "
            f"Expected V1.1.0 capitalised schema. "
            f"Got columns: {list(df.columns)}"
        )

    # Parse and timezone-normalise the Date column.
    #
    # The V1.1.0 fetcher writes tz-aware IST timestamps like
    # '2026-05-08 00:00:00+05:30'. Pandas parses these into a tz-aware
    # datetime64 Series. Downstream code (merge, diagnostics, visualize)
    # builds business-day forecast grids via pd.bdate_range, which are
    # tz-naive. A direct .map() between a tz-naive key and a tz-aware
    # index silently returns NaN for every lookup — this is the root
    # cause of "Actuals filled: 0" symptoms.
    #
    # Fix: strip the timezone if present, keeping the local calendar
    # date as written. tz_localize(None) is correct here (not
    # tz_convert(None) and not utc=True): we want '2026-05-08+05:30'
    # to stay as '2026-05-08', not get shifted to '2026-05-07 18:30'.
    parsed = pd.to_datetime(df["Date"])
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    df["Date"] = parsed.dt.normalize()

    df = df.sort_values("Date").reset_index(drop=True)
    return df


def list_stocks(
    data_dir: Optional[Union[Path, str]] = None,
) -> list[str]:
    """
    List available stock symbols based on CSV files in `data_dir`.

    Parameters
    ----------
    data_dir : Directory containing per-symbol CSVs. Defaults to `./data/`.

    Returns
    -------
    Alphabetically sorted list of symbol strings (CSV filename stems).
    Returns an empty list if the directory doesn't exist.
    """
    base = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.csv"))
