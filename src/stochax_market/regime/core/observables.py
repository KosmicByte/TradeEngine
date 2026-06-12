"""Lifted observables ψ(x) for Extended DMD.

The choice of observables is the heart of EDMD. We use a documented, configurable
dictionary so the analysis is reproducible and the user can tune it.

All functions are pure and operate on numpy arrays — pass in a close-price series,
get back a feature matrix shaped (T, k) where k = number of observables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ObservableConfig:
    """Switches controlling which observables are built."""

    use_log_return: bool = True
    use_squared_return: bool = True
    use_abs_return: bool = True
    momentum_windows: tuple[int, ...] = field(default_factory=lambda: (5, 21, 63))
    vol_windows: tuple[int, ...] = field(default_factory=lambda: (5, 21, 63))
    use_drawdown: bool = True
    # Price-level z-score windows: log-price minus its rolling mean, normalised by
    # rolling stdev. This is critical for detecting mean-reversion at the PRICE level
    # — pure return-based observables can't see OU-style reversion.
    price_zscore_windows: tuple[int, ...] = field(default_factory=lambda: (21, 63))


def log_returns(close: np.ndarray) -> np.ndarray:
    """log(close_t / close_{t-1}). Length T-1; first bar dropped."""
    return np.diff(np.log(np.asarray(close, dtype=float)))


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling stdev with NaN-padding at the start."""
    if window < 2:
        raise ValueError("window must be >= 2")
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if n < window:
        return out
    # cumulative sums for O(n) rolling std
    c1 = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x * x, 0, 0.0))
    sums = c1[window:] - c1[:-window]
    sumsq = c2[window:] - c2[:-window]
    mean = sums / window
    var = np.maximum(sumsq / window - mean * mean, 0.0)
    out[window - 1 :] = np.sqrt(var)
    return out


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean with NaN-padding at the start."""
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if n < window:
        return out
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[window - 1 :] = (c[window:] - c[:-window]) / window
    return out


def drawdown(close: np.ndarray) -> np.ndarray:
    """Drawdown from running maximum, in [-1, 0]."""
    peak = np.maximum.accumulate(close)
    return close / peak - 1.0


def build_observables(
    close: np.ndarray,
    config: ObservableConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build the lifted feature matrix Ψ.

    Args:
        close: 1-D array of close prices, length T.
        config: which observables to include.

    Returns:
        (Psi, names) where Psi has shape (T_valid, k). Rows containing any NaN are dropped.
    """
    config = config or ObservableConfig()
    close = np.asarray(close, dtype=float)
    if close.ndim != 1 or len(close) < 2:
        raise ValueError("close must be a 1-D array with at least 2 points")

    # Align everything to length T-1 by computing returns first.
    r = log_returns(close)  # length T-1
    close_aligned = close[1:]
    cols: list[np.ndarray] = []
    names: list[str] = []

    if config.use_log_return:
        cols.append(r)
        names.append("log_return")
    if config.use_squared_return:
        cols.append(r * r)
        names.append("squared_return")
    if config.use_abs_return:
        cols.append(np.abs(r))
        names.append("abs_return")

    for w in config.momentum_windows:
        cols.append(rolling_mean(r, w))
        names.append(f"momentum_{w}")
    for w in config.vol_windows:
        cols.append(rolling_std(r, w))
        names.append(f"vol_{w}")

    if config.use_drawdown:
        cols.append(drawdown(close_aligned))
        names.append("drawdown")

    # Log-price z-score against a rolling window — visible mean-reversion signal.
    log_close = np.log(close_aligned)
    for w in config.price_zscore_windows:
        rm = rolling_mean(log_close, w)
        rs = rolling_std(log_close, w)
        # Avoid division by zero: replace tiny stds with NaN, the row gets dropped later.
        rs_safe = np.where(rs < 1e-10, np.nan, rs)
        z = (log_close - rm) / rs_safe
        cols.append(z)
        names.append(f"price_zscore_{w}")

    psi = np.column_stack(cols)
    # Drop any row with NaN (the warm-up region of rolling windows).
    valid = ~np.isnan(psi).any(axis=1)
    return psi[valid], names


def standardise(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-mean, unit-variance per column. Returns (psi_std, mean, std)."""
    mean = psi.mean(axis=0)
    std = psi.std(axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    return (psi - mean) / std_safe, mean, std
