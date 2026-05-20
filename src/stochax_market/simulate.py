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
from stochax_market.model.spde import SPDEStepper
from stochax_market.model.volatility import GARCHVolatility


def _load_params(params_path: Path) -> tuple[SPDEStepper, GARCHVolatility, float]:
    """Load fitted params.pkl, handling both tuple and dict formats."""
    with open(params_path, "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, tuple) and len(raw) == 3:
        return raw  # (SPDEStepper, GARCHVolatility, L)
    if isinstance(raw, dict):
        return raw["spde"], raw["garch"], float(raw["L"])
    raise TypeError(f"Unrecognised params format: {type(raw)}")


def simulate(
    symbol: str,
    n_steps: int = 252,
    output_path: str | None = None,
    params_path: str | None = None,
    seed: int = 42,
    nx: int = 128,
    drift_window: int | None = None,
) -> dict:
    """Load stock data, build features, run SPDE, save results.

    Single deterministic-seed forward sample path of the SPDE rolled out
    from the last historical price. For Monte Carlo aggregation with
    confidence bands, use ``stochax_market.predict.predict``.

    Forecast drift
    --------------
    The forward-rollout drift is set to a single scalar — either the full
    historical mean drift (default) or the mean over the most recent
    ``drift_window`` days. This matches ``predict.py``'s behaviour.

    The earlier (pre-v0.9) version of this function passed
    ``features["drift"][-nt:]`` directly to ``spde.rollout``, which fed the
    raw historical log-return *sequence* in as forward drift — i.e. it
    deterministically replayed the last nt days of price history as if
    they were the forecast. The August 2026 cliff dive observed in the
    RELIANCE v0.8 backtest was exactly this artefact: the last 150 days
    of training data (Oct 2025 → Mar 2026) contained RELIANCE's peak and
    subsequent drawdown, so the "forecast" reproduced the drawdown in
    August 2026.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE').
        n_steps: Number of timesteps to simulate.
        output_path: Optional path for output CSV. Defaults to
            '{symbol}_sim.csv'.
        params_path: Optional path to fitted params.pkl. Uses default init
            if None.
        seed: Random seed.
        nx: Number of spatial grid points.
        drift_window: If set, mean the historical drift only over the last
            ``drift_window`` days. Default = None → full-history mean.
            Useful when the diagnostics report flags a regime shift (e.g.
            "recent-60d drift diverges from full-history mean"); try 126
            (~6 months) or 252 (~1 year) to track the recent trend.

    Returns:
        Dict with keys: predicted_prices, trajectory_shape, output_path,
        n_steps.
    """
    df       = load_stock(symbol)
    features = encode_features(df, nx=nx)

    T  = features["log_returns"].shape[0]
    nt = min(n_steps, T)

    key = jax.random.key(seed)

    # ── Load fitted params if provided, else use defaults ─────────────────────
    if params_path is not None and Path(params_path).exists():
        spde, garch, L = _load_params(Path(params_path))
    else:
        spde  = SPDEStepper(nx=nx)
        garch = GARCHVolatility()
        L     = float(features.get("L", features.get("price_scale", 1.0)))

    # ── Build noise and volatility fields ─────────────────────────────────────
    # σ trajectory: GARCH applied to the most recent nt log-returns. This
    # gives a vol path with realistic recent-history characteristics; for a
    # fully stochastic forward GARCH simulation, use predict.py.
    sigma_series = garch(features["log_returns"][-nt:])
    sigma_fields = jax.vmap(GARCHVolatility.to_spatial_field, in_axes=(0, None))(
        sigma_series, nx
    )

    key, noise_key = jax.random.split(key)
    noise = make_noise_trajectory(noise_key, nt, nx, spde.dt)

    # ── Forecast drift: scalar mean (windowed if requested) ─────────────────
    # See docstring for the rationale and the v0.9 bug-fix history.
    if drift_window is None or drift_window >= features["drift"].shape[0]:
        mean_drift = jnp.mean(features["drift"])
    else:
        mean_drift = jnp.mean(features["drift"][-drift_window:])
    drift_forward = jnp.full(nt, mean_drift, dtype=jnp.float32)

    # ── Initial condition anchored at last known price ────────────────────────
    # make_initial_condition normalises last_price by domain_extent internally.
    last_price = float(df["Close"].iloc[-1])
    u0_start = make_initial_condition(last_price, domain_extent=L, nx=nx)

    trajectory = spde.rollout(
        u0_start, sigma_fields, noise, drift_forward, nt
    )

    # ── Extract prices: field_mean returns value in [0, domain_extent] ───────
    # Multiply by L once to recover INR.
    x_grid    = jnp.linspace(0.0, spde.domain_extent, nx, dtype=jnp.float32)
    predicted = jax.vmap(field_mean, in_axes=(0, None))(trajectory, x_grid)
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
