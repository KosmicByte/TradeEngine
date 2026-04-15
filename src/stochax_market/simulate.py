"""Simulate SPDE forward in time for a given stock symbol."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp

from stochax_market.calibration.loss import field_mean
from stochax_market.data.features import encode_features
from stochax_market.data.loader import load_stock
from stochax_market.model.initial import make_initial_condition
from stochax_market.model.noise import make_noise_trajectory
from stochax_market.model.spde import PIDController, SPDEStepper
from stochax_market.model.volatility import GARCHVolatility


def _load_params(
    params_path: Path,
) -> tuple[SPDEStepper, GARCHVolatility, PIDController, float]:
    """Load fitted params.pkl, handling both tuple and dict formats."""
    with open(params_path, "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, dict):
        spde  = raw["spde"]
        garch = raw["garch"]
        pid   = raw.get("pid", PIDController())
        L     = float(raw["L"])
        return spde, garch, pid, L
    if isinstance(raw, tuple) and len(raw) == 3:
        spde, garch, L = raw
        return spde, garch, PIDController(), L
    raise TypeError(f"Unrecognised params format: {type(raw)}")


def simulate(
    symbol: str,
    n_steps: int = 252,
    output_path: str | None = None,
    params_path: str | None = None,
    seed: int = 42,
    nx: int = 128,
) -> dict:
    """Load stock data, build features, run SPDE, save results.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE').
        n_steps: Number of timesteps to simulate.
        output_path: Optional path for output CSV. Defaults to '{symbol}_sim.csv'.
        params_path: Optional path to fitted params.pkl. Uses default init if None.
        seed: Random seed.
        nx: Number of spatial grid points.

    Returns:
        Dict with keys: predicted_prices, trajectory shape, output_path.
    """
    df       = load_stock(symbol)
    features = encode_features(df, nx=nx)

    T  = features["log_returns"].shape[0]
    nt = min(n_steps, T)

    key = jax.random.key(seed)

    # ── Load fitted params if provided, else use defaults ─────────────────────
    if params_path is not None and Path(params_path).exists():
        spde, garch, pid, L = _load_params(Path(params_path))
    else:
        spde  = SPDEStepper(nx=nx)
        garch = GARCHVolatility()
        pid   = PIDController()
        L     = float(features.get("L", features.get("price_scale", 1.0)))

    # ── Build noise and volatility fields ─────────────────────────────────────
    sigma_series = garch(features["log_returns"][-nt:])
    sigma_fields = jax.vmap(GARCHVolatility.to_spatial_field, in_axes=(0, None))(
        sigma_series, nx
    )

    key, noise_key = jax.random.split(key)
    noise = make_noise_trajectory(noise_key, nt, nx, spde.dt)

    # ── Initial condition anchored at last known price ─────────────────────────
    last_price = float(df["Close"].iloc[-1])
    u0_start   = make_initial_condition(last_price, domain_extent=L, nx=nx)

    trajectory = spde.rollout(
        u0_start, sigma_fields, noise, features["drift"][-nt:], nt,
        pid=pid, setpoint_series=features["vwap_target"][-nt:],
    )

    # ── Extract prices: field_mean returns normalised coordinate → scale to INR ─
    x_grid    = jnp.linspace(0.0, spde.domain_extent, nx, dtype=jnp.float32)
    predicted = jax.vmap(field_mean, in_axes=(0, None))(trajectory, x_grid)
    # field_mean returns value in [0, domain_extent]; multiply once by L for INR
    predicted_prices = predicted * L

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
        "output_path":      str(out),
        "n_steps":          nt,
    }