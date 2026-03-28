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

# Maximum number of BFGS optimisation steps for calibration.
_MAX_CALIBRATION_STEPS: int = 1000

# Residuals larger than this magnitude are clipped to prevent NaN from
# stiff SPDE steps propagating into the BFGS gradient update.
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

    Extracts trainable float parameters (mu_scale, omega, alpha, beta) into
    a flat array, keeping non-differentiable components (exponax stepper)
    frozen. Uses optimistix.minimise with BFGS.

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

    # Limit data size for tractable optimization
    max_T = min(data["log_returns"].shape[0], 50)
    small_data = {
        k: v[:max_T] if hasattr(v, "shape") and len(v.shape) > 0 else v
        for k, v in data.items()
    }

    # Extract trainable parameters as a flat array to avoid
    # differentiating through exponax's non-differentiable stepper
    y0 = jnp.array([
        model_spde.raw_mu_scale,
        model_garch.raw_omega,
        model_garch.raw_alpha,
        model_garch.raw_beta,
    ], dtype=jnp.float32)

    # Frozen model templates for reconstruction
    frozen_spde = model_spde
    frozen_garch = model_garch

    def loss_fn(y, args):
        noise_key = args
        # Reconstruct models from flat parameter array
        spde = eqx.tree_at(lambda s: s.raw_mu_scale, frozen_spde, y[0])
        garch = eqx.tree_at(
            lambda g: (g.raw_omega, g.raw_alpha, g.raw_beta),
            frozen_garch,
            (y[1], y[2], y[3]),
        )

        # Penalty: reward non-constant volatility by maximising the masked
        # variance of the GARCH sigma trajectory.  Boolean indexing produces
        # dynamic-length arrays that break JAX JIT/grad; instead we keep the
        # shape static by using jnp.where and computing a weighted variance.
        lr = small_data["log_returns"]
        lr_mask = (~jnp.isnan(lr)).astype(jnp.float32)
        lr_safe = jnp.where(jnp.isnan(lr), 0.0, lr)
        sigma_trajectory = garch(lr_safe)
        count = jnp.sum(lr_mask)
        safe_count = jnp.maximum(count, 1.0)
        sigma_mean = jnp.sum(sigma_trajectory * lr_mask) / safe_count
        # Population variance (÷ N) is appropriate here: this is a regularisation
        # penalty, not a statistical estimator, so Bessel's correction is not needed.
        sigma_var = jnp.sum(lr_mask * (sigma_trajectory - sigma_mean) ** 2) / safe_count
        sigma_var_masked = jnp.where(count > 1.0, sigma_var, 0.0)
        sigma_variance_penalty = -0.01 * sigma_var_masked

        predicted = _run_model(spde, garch, small_data, noise_key)
        target = small_data["close_target"]
        n = min(predicted.shape[0], target.shape[0])
        # Clip residuals to prevent NaN from stiff SPDE steps
        residuals = jnp.clip(
            predicted[:n] - target[:n],
            -_MAX_RESIDUAL_MAGNITUDE,
            _MAX_RESIDUAL_MAGNITUDE,
        )
        mse = jnp.mean(residuals ** 2)
        return mse + sigma_variance_penalty

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
        y_opt = sol.value
        final_loss = float(loss_fn(y_opt, key))

        # Reconstruct fitted models
        fitted_spde = eqx.tree_at(
            lambda s: s.raw_mu_scale, model_spde, y_opt[0]
        )
        fitted_garch = eqx.tree_at(
            lambda g: (g.raw_omega, g.raw_alpha, g.raw_beta),
            model_garch,
            (y_opt[1], y_opt[2], y_opt[3]),
        )
        result_info = {
            "steps": _MAX_CALIBRATION_STEPS,
            "result": str(sol.result),
            "final_loss": final_loss,
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
