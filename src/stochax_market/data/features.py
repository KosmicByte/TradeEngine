"""Feature engineering for SPDE-based stock price modelling."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd

from stochax_market.model.initial import price_to_field

_REQUIRED_COLUMNS = {"Date", "Prev Close", "Open", "High", "Low", "Close", "VWAP"}


def encode_features(df: pd.DataFrame, nx: int = 128) -> dict:
    """Build feature arrays for the SPDE model from an OHLCV DataFrame.

    Validates required columns, drops rows with missing ``Prev Close`` or
    ``Close`` values, normalises prices to the SPDE domain ``[0, 1]``, and
    returns all arrays needed by :func:`stochax_market.simulate.simulate` and
    :func:`stochax_market.predict.predict`.

    Args:
        df: DataFrame with columns ``Date``, ``Prev Close``, ``Open``,
            ``High``, ``Low``, ``Close``, ``VWAP``.
        nx: Number of spatial grid points for the SPDE (default: 128).

    Returns:
        Dict with keys:

        - ``u0`` (T, nx): Gaussian initial conditions centred at the
          normalised previous-close price at each timestep.
        - ``drift`` (T,): Log-return drift series used as the per-step drift
          input to the SPDE stepper.
        - ``log_returns`` (T,): Log returns ``log(Close / Prev Close)``.
        - ``vwap_target`` (T,): Normalised VWAP prices.
        - ``close_target`` (T,): Normalised Close prices.
        - ``price_range`` (T, 2): Normalised ``[Low, High]`` at each timestep.
        - ``dates`` (T,): Date values from the ``Date`` column.
        - ``seasonal`` (T, 2): ``[sin, cos]`` of day-of-year seasonality.
        - ``price_scale`` (float): Scalar to recover original prices from
          normalised values.

    Raises:
        ValueError: If any required column is missing from *df*.
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    # Drop rows where the essential price columns are NaN (e.g. rolled first row)
    df = df.dropna(subset=["Prev Close", "Close"]).reset_index(drop=True)

    close = df["Close"].to_numpy(dtype=np.float64)
    prev_close = df["Prev Close"].to_numpy(dtype=np.float64)
    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)
    vwap = df["VWAP"].to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Price normalisation: scale all prices to [0, 1] so they fit within  #
    # the SPDE domain [0, domain_extent=1].                               #
    # ------------------------------------------------------------------ #
    price_scale = float(
        np.max(np.abs(np.concatenate([close, prev_close, high, low, vwap]))) + 1e-8
    )

    close_norm = (close / price_scale).astype(np.float32)
    prev_close_norm = (prev_close / price_scale).astype(np.float32)
    high_norm = (high / price_scale).astype(np.float32)
    low_norm = (low / price_scale).astype(np.float32)
    vwap_norm = (vwap / price_scale).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Log returns and drift                                               #
    # ------------------------------------------------------------------ #
    log_returns = np.log(close / prev_close).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Seasonal features: day-of-year encoded as (sin, cos) pair           #
    # ------------------------------------------------------------------ #
    dates = df["Date"].to_numpy()
    day_of_year = pd.DatetimeIndex(df["Date"]).day_of_year.to_numpy().astype(np.float32)
    seasonal = np.stack(
        [
            np.sin(2.0 * np.pi * day_of_year / 365.0),
            np.cos(2.0 * np.pi * day_of_year / 365.0),
        ],
        axis=1,
    ).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Initial conditions: Gaussian blob centred at normalised Prev Close   #
    # ------------------------------------------------------------------ #
    u0 = jnp.stack(
        [price_to_field(float(p), L=1.0, nx=nx) for p in prev_close_norm]
    )  # (T, nx)

    price_range = jnp.stack(
        [
            jnp.array(low_norm, dtype=jnp.float32),
            jnp.array(high_norm, dtype=jnp.float32),
        ],
        axis=1,
    )  # (T, 2) — columns: [Low, High]

    return {
        "u0": u0,
        "drift": jnp.array(log_returns, dtype=jnp.float32),
        "log_returns": jnp.array(log_returns, dtype=jnp.float32),
        "vwap_target": jnp.array(vwap_norm, dtype=jnp.float32),
        "close_target": jnp.array(close_norm, dtype=jnp.float32),
        "price_range": price_range,
        "dates": dates,
        "seasonal": jnp.array(seasonal, dtype=jnp.float32),
        "price_scale": price_scale,
    }
