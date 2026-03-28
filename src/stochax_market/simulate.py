"""Simulate SPDE forward in time for a given stock symbol."""

from __future__ import annotations

import csv
from pathlib import Path

import jax
import jax.numpy as jnp

from stochax_market.calibration.loss import field_mean
from stochax_market.data.features import encode_features
from stochax_market.data.loader import load_stock
from stochax_market.model.noise import make_noise_trajectory
from stochax_market.model.spde import SPDEStepper
from stochax_market.model.volatility import GARCHVolatility


def simulate(
    symbol: str,
    n_steps: int = 252,
    output_path: str | None = None,
    seed: int = 42,
    nx: int = 128,
) -> dict:
    """Load stock data, build features, run SPDE, save results.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE').
        n_steps: Number of timesteps to simulate.
        output_path: Optional path for output CSV. Defaults to '{symbol}_sim.csv'.
        seed: Random seed.
        nx: Number of spatial grid points.

    Returns:
        Dict with keys: predicted_prices, trajectory shape, output_path.
    """
    df = load_stock(symbol)
    features = encode_features(df, nx=nx)

    T = features["log_returns"].shape[0]
    nt = min(n_steps, T)

    key = jax.random.key(seed)

    spde = SPDEStepper(nx=nx)
    garch = GARCHVolatility()

    sigma_series = garch(features["log_returns"][:nt])
    sigma_fields = jax.vmap(GARCHVolatility.to_spatial_field, in_axes=(0, None))(
        sigma_series, nx
    )

    key, noise_key = jax.random.split(key)
    noise = make_noise_trajectory(noise_key, nt, nx, spde.dt)

    trajectory = spde.rollout(
        features["u0"][0], sigma_fields, noise, features["drift"][:nt], nt
    )

    x_grid = jnp.linspace(0.0, spde.domain_extent, nx, dtype=jnp.float32)
    predicted = jax.vmap(field_mean, in_axes=(0, None))(trajectory, x_grid)

    price_scale = features.get("L", features.get("price_scale", 1.0))
    predicted_prices = predicted * price_scale

    if output_path is None:
        output_path = f"{symbol}_sim.csv"

    out = Path(output_path)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "predicted_price"])
        for i, p in enumerate(predicted_prices.tolist()):
            writer.writerow([i, f"{p:.4f}"])

    return {
        "predicted_prices": predicted_prices,
        "trajectory_shape": trajectory.shape,
        "output_path": str(out),
        "n_steps": nt,
    }
