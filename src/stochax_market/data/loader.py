"""Load stock price data from local CSV files or Kaggle."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Kaggle dataset that provides per-symbol CSV files matching the expected schema:
# Date, Prev Close, Open, High, Low, Last, Close, VWAP, Volume, Turnover, Trades,
# Deliverable Volume, %Deliverable
_KAGGLE_DATASET = "rohanrao/nifty50-stock-market-data"

# Default local data directory: <repo-root>/data/
# This module lives at src/stochax_market/data/loader.py, so parents[3] is the repo root.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def load_stock(symbol: str, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load historical OHLCV data for a stock symbol.

    Looks for ``<SYMBOL>.csv`` in the local data directory first.  If not
    found, downloads the NIFTY50 dataset from Kaggle via ``kagglehub`` and
    reads the file from the cached download.

    The returned DataFrame is sorted by ``Date`` ascending and has at minimum
    the columns required by :func:`stochax_market.data.features.encode_features`:
    ``Date``, ``Prev Close``, ``Open``, ``High``, ``Low``, ``Close``, ``VWAP``.

    Args:
        symbol: Stock ticker symbol, e.g. ``'RELIANCE'``.
        data_dir: Directory to search for local CSV files.  Defaults to the
            ``data/`` folder at the repository root.

    Returns:
        DataFrame sorted by Date ascending.

    Raises:
        FileNotFoundError: If the symbol cannot be found locally or on Kaggle.
    """
    data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
    local_path = data_dir / f"{symbol}.csv"

    if local_path.exists():
        df = pd.read_csv(local_path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df

    # Fall back to kagglehub download
    try:
        import kagglehub  # type: ignore[import-untyped]

        dataset_path = Path(kagglehub.dataset_download(_KAGGLE_DATASET))
        csv_path = dataset_path / f"{symbol}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Symbol '{symbol}' not found in Kaggle dataset at {dataset_path}. "
                f"Available files: {[p.stem for p in dataset_path.glob('*.csv')]}"
            )
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df
    except ImportError as exc:
        raise FileNotFoundError(
            f"Could not load data for '{symbol}': '{local_path}' does not exist "
            f"and 'kagglehub' is not installed."
        ) from exc
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not load data for '{symbol}'. "
            f"Place '{symbol}.csv' in '{data_dir}' or configure Kaggle credentials. "
            f"Original error: {exc}"
        ) from exc


def list_stocks(data_dir: str | Path | None = None) -> list[str]:
    """List available stock symbols.

    Returns symbols found in the local data directory.  If the local directory
    does not exist or is empty, falls back to listing symbols in the Kaggle
    dataset cache (downloading if necessary).

    Args:
        data_dir: Directory to search for local CSV files.  Defaults to the
            ``data/`` folder at the repository root.

    Returns:
        Sorted list of symbol strings (without the ``.csv`` extension).
    """
    data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR

    local_csvs = sorted(p.stem for p in data_dir.glob("*.csv")) if data_dir.exists() else []
    if local_csvs:
        return local_csvs

    try:
        import kagglehub  # type: ignore[import-untyped]

        dataset_path = Path(kagglehub.dataset_download(_KAGGLE_DATASET))
        return sorted(p.stem for p in dataset_path.glob("*.csv"))
    except Exception:
        return []
