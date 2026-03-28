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
          normalised values. Equals ``L``.
        - ``L`` (float): Spatial domain length = ``max(High)`` across the
          full dataset.  Use ``predicted_price = field_mean * L`` to
          denormalise model outputs.

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
    #                                                                      #
    # L is the spatial domain length, anchored to the full-dataset High   #
    # maximum so that the domain is stable across fit / simulate / predict #
    # calls and is not accidentally recomputed from a data batch.          #
    # ------------------------------------------------------------------ #
    L = float(np.max(high))
    price_scale = L  # kept for backward compatibility

    close_norm = (close / L).astype(np.float32)
    prev_close_norm = (prev_close / L).astype(np.float32)
    high_norm = (high / L).astype(np.float32)
    low_norm = (low / L).astype(np.float32)
    vwap_norm = (vwap / L).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Log returns and drift                                               #
    # Split/bonus-issue events produce single-day returns of ±50–70 %,  #
    # which corrupt GARCH calibration.  Any |log_return| > 0.25 is      #
    # almost certainly a corporate action, not a real price move.        #
    # Replace those days with 0.0 so GARCH sees a zero-return (neutral) #
    # and the SPDE drift term also receives a safe finite value.         #
    # Using 0.0 (not NaN) keeps all downstream arrays NaN-free and      #
    # prevents NaN propagation through SPDEStepper.step().               #
    # ------------------------------------------------------------------ #
    _SPLIT_THRESHOLD = 0.25
    log_returns = np.log(close / prev_close).astype(np.float32)
    split_mask = np.abs(log_returns) > _SPLIT_THRESHOLD
    log_returns[split_mask] = 0.0

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
    # Prices are normalised to [0, 1] (divided by L), so the grid domain  #
    # passed to price_to_field is L=1.0 (the unit-normalised interval).   #
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
        "L": L,
        "price_scale": price_scale,
    }
