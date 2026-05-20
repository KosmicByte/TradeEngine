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
    drift_window: int | None = None,
) -> dict[str, Any]:
    """Forecast Close h steps ahead with bootstrap confidence intervals.

    Runs K=n_samples Monte Carlo forward simulations using jax.vmap over seeds
    to produce predicted prices with confidence intervals.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE').
        horizon: Number of steps ahead to forecast.
        params_path: Path to pickled (spde, garch) params. If None, uses defaults.
        seed: Base random seed.
        n_samples: Number of Monte Carlo samples for confidence intervals.
        nx: Number of spatial grid points.
        drift_window: If set, use only the last `drift_window` historical
            timesteps to compute the mean drift used in the forecast. If
            None (default), uses the full-history mean. A shorter window
            (e.g. 60–126 trading days) tracks recent regime shifts; the
            full-history mean is more stable but lags. See the regime-shift
            note below.

    Returns:
        Dict with keys: mean_prediction, lower_ci, upper_ci, all_samples.

    Notes
    -----
    Two forecasting bugs were patched in this version. Their previous (buggy)
    forms are preserved in the docstring of `_single_forecast` below.

    Regime-shift handling (drift_window):
        The diagnostics module compares full-history drift against recent-60d
        drift. When the two diverge by more than ~30% annualised, the
        full-history mean injects a directional bias that systematically
        over- or under-shoots the next month. For RELIANCE on 2026-05-08,
        historical drift annualised to 15.8% but recent-60d to −2.4%,
        producing a +3.6% bias. Passing `drift_window=126` (≈6 months)
        damps the historical bullish prior toward the recent regime.
    """
    df = load_stock(symbol)
    features = encode_features(df, nx=nx)

    if params_path is not None and Path(params_path).exists():
        with open(params_path, "rb") as f:
            saved = pickle.load(f)
        if isinstance(saved, dict):
            spde = saved["spde"]
            garch = saved["garch"]
            L = float(saved["L"])
        elif len(saved) == 3:
            spde, garch, L = saved
        else:
            spde, garch = saved
            L = None

    # Fall back to the L computed from the current dataset
    if L is None:
        L = features.get("L", features.get("price_scale", 1.0))

    log_returns = features["log_returns"]
    sigma_series = garch(log_returns)

    last_u     = features["u0"][-1]
    last_sigma = sigma_series[-1]
    last_eps   = log_returns[-1]                  # last innovation; seeds the GARCH forecast

    # Drift baseline: configurable window. Default → full history.
    if drift_window is None or drift_window >= features["drift"].shape[0]:
        mean_drift = jnp.mean(features["drift"])
    else:
        mean_drift = jnp.mean(features["drift"][-drift_window:])

    # Previously (buggy):
    #     last_drift = features["drift"][-1]
    # See _single_forecast docstring for the rationale.

    def _single_forecast(key: jax.Array) -> jnp.ndarray:
        """Roll forward one Monte Carlo path of length `horizon`.

        Patches applied
        ---------------

        Bug 1 — Frozen volatility (the GARCH dynamics never fired):

            # OLD (buggy):
            sigma_fields = jax.vmap(
                GARCHVolatility.to_spatial_field, in_axes=(0, None)
            )(jnp.full(horizon, last_sigma), nx)

        `last_sigma` was held constant for every step of the forecast, so the
        fitted GARCH(1,1) coefficients (α=0.1445, β=0.8280) had no effect on
        the forward path — the whole point of the volatility model was lost.
        We now iterate the GARCH recursion forward stochastically, sampling a
        fresh innovation ε_t ~ N(0, σ²_t) at each step and propagating
        σ²_{t+1} = ω + α ε²_t + β σ²_t. Each Monte Carlo sample now carries
        its own σ trajectory, so the 5/95 CI band fans out with horizon as it
        physically should.

        Bug 2 — Constant drift (the directional collapse):

            # OLD (buggy):
            drift = jnp.full(horizon, last_drift)

        `last_drift` was the log-return on the final historical day — a single
        noisy point. Broadcasting it across the whole horizon baked that one
        tick in as a permanent directional bias, producing the flat-to-monotonic
        predicted curves observed in the April-18 RELIANCE backtest. We now
        use a (windowed) mean drift as the deterministic μ-component;
        directional uncertainty is carried by the noise + volatility terms,
        which is where it belongs in this SPDE.
        """
        eps_key, noise_key = jax.random.split(key)

        # ── 1. GARCH(1,1) rolled forward stochastically ───────────────────
        # Mirrors the recursion in volatility.py::GARCHVolatility.__call__,
        # including the σ² ≥ 1e-8 floor.
        eps_z = jax.random.normal(eps_key, (horizon,))

        def garch_step(carry, z_t):
            sigma_sq_prev, eps_prev = carry
            sigma_sq_t = (
                garch.omega
                + garch.alpha * eps_prev ** 2
                + garch.beta  * sigma_sq_prev
            )
            sigma_sq_t = jnp.maximum(sigma_sq_t, 1e-8)
            sigma_t    = jnp.sqrt(sigma_sq_t)
            eps_t      = z_t * sigma_t
            return (sigma_sq_t, eps_t), sigma_t

        init = (last_sigma ** 2, last_eps)
        _, sigma_path = jax.lax.scan(garch_step, init, eps_z)   # (horizon,)

        sigma_fields = jax.vmap(
            GARCHVolatility.to_spatial_field, in_axes=(0, None)
        )(sigma_path, nx)

        # ── 2. Drift: mean over the chosen window (deterministic μ component) ─
        drift = jnp.full(horizon, mean_drift)

        # ── 3. Spatial Q-Wiener noise (unchanged) ─────────────────────────
        noise = make_noise_trajectory(noise_key, horizon, nx, spde.dt)

        traj = spde.rollout(last_u, sigma_fields, noise, drift, horizon)

        x_grid = jnp.linspace(0.0, spde.domain_extent, nx, dtype=jnp.float32)
        prices = jax.vmap(field_mean, in_axes=(0, None))(traj, x_grid)
        return prices * L

    keys = jax.random.split(jax.random.key(seed), n_samples)
    all_samples = jax.vmap(_single_forecast)(keys)  # (n_samples, horizon)

    mean_pred = jnp.mean(all_samples, axis=0)
    lower_ci  = jnp.percentile(all_samples, 5.0, axis=0)
    upper_ci  = jnp.percentile(all_samples, 95.0, axis=0)

    return {
        "mean_prediction": mean_pred,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "all_samples": all_samples,
        "horizon": horizon,
        "n_samples": n_samples,
    }
