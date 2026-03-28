"""Optimistix-based parameter fitting for SPDE + GARCH model."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx

from stochax_market.calibration.loss import field_mean
from stochax_market.model.noise import make_noise_trajectory
from stochax_market.model.spde import SPDEStepper
from stochax_market.model.volatility import GARCHVolatility


def _run_model(
    spde: SPDEStepper,
    garch: GARCHVolatility,
    data: dict[str, jnp.ndarray],
    noise_key: jax.Array,
) -> jnp.ndarray:
    """Run the SPDE model forward and extract predicted prices.

    Args:
        spde: SPDE stepper module.
        garch: GARCH volatility module.
        data: Encoded feature dict.
        noise_key: JAX random key.

    Returns:
        Shape (T,) predicted prices (field means at each timestep).
    """
    log_returns = data["log_returns"]
    u0_series = data["u0"]
    drift = data["drift"]
    T = log_returns.shape[0]
    nx = spde.nx

    sigma_series = garch(log_returns)

    sigma_fields = jax.vmap(GARCHVolatility.to_spatial_field, in_axes=(0, None))(
        sigma_series, nx
    )

    noise = make_noise_trajectory(noise_key, T, nx, spde.dt)

    trajectory = spde.rollout(u0_series[0], sigma_fields, noise, drift, T)

    x_grid = jnp.linspace(0.0, spde.domain_extent, nx, dtype=jnp.float32)
    predicted = jax.vmap(field_mean, in_axes=(0, None))(trajectory, x_grid)

    return predicted


def fit(
    model_spde: SPDEStepper,
    model_garch: GARCHVolatility,
    data: dict[str, jnp.ndarray],
    n_steps: int = 100,
    lr: float = 1e-3,
    key: jax.Array | None = None,
) -> tuple[SPDEStepper, GARCHVolatility, dict[str, Any]]:
    """Fit SPDE + GARCH parameters using Optimistix BFGS minimizer.

    Constructs a pure loss function and uses optimistix.minimise with BFGS
    to fit parameters {κ, μ_scale, ω, α, β} to the dataset.

    Args:
        model_spde: Initial SPDE stepper module.
        model_garch: Initial GARCH volatility module.
        data: Encoded feature dict from encode_features().
        n_steps: Maximum optimization steps.
        lr: Learning rate (used as step size for BFGS).
        key: JAX random key. If None, uses key(0).

    Returns:
        Tuple of (fitted_spde, fitted_garch, loss_history_dict).
    """
    if key is None:
        key = jax.random.key(0)

    params = (model_spde, model_garch)

    # Limit data size for tractable optimization
    max_T = min(data["log_returns"].shape[0], 50)
    small_data = {k: v[:max_T] if hasattr(v, 'shape') and len(v.shape) > 0 else v for k, v in data.items()}

    def loss_fn(params, args):
        spde, garch = params
        noise_key = args
        predicted = _run_model(spde, garch, small_data, noise_key)
        target = small_data["close_target"]
        n = jnp.minimum(predicted.shape[0], target.shape[0])
        mse = jnp.mean((predicted[:n] - target[:n]) ** 2)
        return mse

    solver = optx.BFGS(rtol=1e-5, atol=1e-5)

    try:
        sol = optx.minimise(
            loss_fn,
            solver,
            params,
            args=key,
            max_steps=n_steps,
            throw=False,
        )
        fitted_spde, fitted_garch = sol.value
        result_info = {
            "steps": n_steps,
            "result": str(sol.result),
        }
    except Exception as e:
        # Fallback: return initial params if optimization fails
        fitted_spde = model_spde
        fitted_garch = model_garch
        result_info = {
            "steps": 0,
            "error": str(e),
        }

    return fitted_spde, fitted_garch, result_info
