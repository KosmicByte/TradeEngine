"""Predict future stock prices using fitted SPDE + GARCH model."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from stochax_market.calibration.loss import field_mean
from stochax_market.data.features import encode_features
from stochax_market.data.loader import load_stock
from stochax_market.model.noise import make_noise_trajectory
from stochax_market.model.spde import SPDEStepper
from stochax_market.model.volatility import GARCHVolatility


def predict(
    symbol: str,
    horizon: int = 5,
    params_path: str | None = None,
    seed: int = 42,
    n_samples: int = 100,
    nx: int = 128,
) -> dict[str, Any]:
    """Forecast Close and VWAP h steps ahead with bootstrap confidence intervals.

    Runs K=n_samples Monte Carlo forward simulations using jax.vmap over seeds
    to produce predicted prices with confidence intervals.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE').
        horizon: Number of steps ahead to forecast.
        params_path: Path to pickled (spde, garch) params. If None, uses defaults.
        seed: Base random seed.
        n_samples: Number of Monte Carlo samples for confidence intervals.
        nx: Number of spatial grid points.

    Returns:
        Dict with keys: mean_prediction, lower_ci, upper_ci, all_samples.
    """
    df = load_stock(symbol)
    features = encode_features(df, nx=nx)

    if params_path is not None and Path(params_path).exists():
        with open(params_path, "rb") as f:
            saved = pickle.load(f)
        if len(saved) == 3:
            spde, garch, L = saved
        else:
            spde, garch = saved
            L = None
    else:
        spde = SPDEStepper(nx=nx)
        garch = GARCHVolatility()
        L = None

    # Fall back to the L computed from the current dataset
    if L is None:
        L = features.get("L", features.get("price_scale", 1.0))

    log_returns = features["log_returns"]
    sigma_series = garch(log_returns)

    last_u = features["u0"][-1]
    last_sigma = sigma_series[-1]
    last_drift = features["drift"][-1]

    def _single_forecast(key: jax.Array) -> jnp.ndarray:
        sigma_fields = jax.vmap(GARCHVolatility.to_spatial_field, in_axes=(0, None))(
            jnp.full(horizon, last_sigma), nx
        )
        noise = make_noise_trajectory(key, horizon, nx, spde.dt)
        drift = jnp.full(horizon, last_drift)

        traj = spde.rollout(last_u, sigma_fields, noise, drift, horizon)

        x_grid = jnp.linspace(0.0, spde.domain_extent, nx, dtype=jnp.float32)
        prices = jax.vmap(field_mean, in_axes=(0, None))(traj, x_grid)
        return prices * L

    keys = jax.random.split(jax.random.key(seed), n_samples)
    all_samples = jax.vmap(_single_forecast)(keys)  # (n_samples, horizon)

    mean_pred = jnp.mean(all_samples, axis=0)
    lower_ci = jnp.percentile(all_samples, 5.0, axis=0)
    upper_ci = jnp.percentile(all_samples, 95.0, axis=0)

    return {
        "mean_prediction": mean_pred,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "all_samples": all_samples,
        "horizon": horizon,
        "n_samples": n_samples,
    }
