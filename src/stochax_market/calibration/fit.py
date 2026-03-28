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

_MAX_CALIBRATION_STEPS: int = 1000
_MAX_RESIDUAL_MAGNITUDE: float = 100.0


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

    Args:
        model_spde: Initial SPDE stepper module.
        model_garch: Initial GARCH volatility module.
        data: Encoded feature dict from encode_features().
        n_steps: Maximum optimization steps.
        lr: Learning rate (unused, BFGS uses line search).
        key: JAX random key. If None, uses key(0).

    Returns:
        Tuple of (fitted_spde, fitted_garch, loss_history_dict).
    """
    if key is None:
        key = jax.random.key(0)

    max_T = min(data["log_returns"].shape[0], 50)
    small_data = {
        k: v[:max_T] if hasattr(v, "shape") and len(v.shape) > 0 else v
        for k, v in data.items()
    }

    y0 = jnp.array([
        model_spde.raw_mu_scale,
        model_garch.raw_omega,
        model_garch.raw_alpha,
        model_garch.raw_beta,
    ], dtype=jnp.float32)

    frozen_spde  = model_spde
    frozen_garch = model_garch

    def loss_fn(y, args):
        noise_key = args

        spde = eqx.tree_at(lambda s: s.raw_mu_scale, frozen_spde, y[0])
        garch = eqx.tree_at(
            lambda g: (g.raw_omega, g.raw_alpha, g.raw_beta),
            frozen_garch,
            (y[1], y[2], y[3]),
        )

        predicted = _run_model(spde, garch, small_data, noise_key)
        target    = small_data["close_target"]
        n         = min(predicted.shape[0], target.shape[0])

        residuals = jnp.clip(
            predicted[:n] - target[:n],
            -_MAX_RESIDUAL_MAGNITUDE,
            _MAX_RESIDUAL_MAGNITUDE,
        )
        return jnp.mean(residuals ** 2)

    solver = optx.BFGS(rtol=1e-5, atol=1e-5)

    try:
        sol = optx.minimise(
            loss_fn,
            solver,
            y0,
            args=key,
            max_steps=_MAX_CALIBRATION_STEPS,
            throw=False,
        )
        y_opt      = sol.value
        final_loss = float(loss_fn(y_opt, key))

        fitted_spde = eqx.tree_at(
            lambda s: s.raw_mu_scale, model_spde, y_opt[0]
        )
        fitted_garch = eqx.tree_at(
            lambda g: (g.raw_omega, g.raw_alpha, g.raw_beta),
            model_garch,
            (y_opt[1], y_opt[2], y_opt[3]),
        )
        result_info = {
            "steps":      _MAX_CALIBRATION_STEPS,
            "result":     str(sol.result),
            "final_loss": final_loss,
        }
    except Exception as e:
        fitted_spde  = model_spde
        fitted_garch = model_garch
        result_info  = {"steps": 0, "error": str(e)}

    return fitted_spde, fitted_garch, result_info