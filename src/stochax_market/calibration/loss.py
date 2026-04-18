"""Loss functions for calibration."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def field_mean(u: jnp.ndarray, x_grid: jnp.ndarray) -> jnp.ndarray:
    """Compute the expected price from a probability-like field.

    Calculates ∫ x·u(x) dx using the trapezoidal rule.

    Args:
        u: Shape (nx,) normalized field (sums to ~1/dx).
        x_grid: Shape (nx,) spatial grid coordinates.

    Returns:
        Scalar expected value (first moment of the field).
    """
    dx = x_grid[1] - x_grid[0]

    """Added  
    mass = jnp.sum(u) * dx
    return jnp.sum(x_grid * u) * dx / (mass + 1e-8) to compute the center of mass (∫ x·u dx / ∫ u dx)
    instead of assuming ∫ u dx = 1. Now even if the field gains or loses mass during rollout,
    the extracted price will still reflect where the peak is on the grid.
       """

    mass = jnp.sum(u) * dx
    return jnp.sum(x_grid * u) * dx / (mass + 1e-8)


def calibration_loss(
    params: dict,
    data_batch: dict[str, jnp.ndarray],
    model_fn: callable,
    noise_key: jax.Array,
) -> jnp.ndarray:
    """Compute calibration loss: MSE + directional penalty.

    MSE between field_mean(u(T)) and close_target across batch, plus
    0.1 * directional penalty (fraction of timesteps where
    sign(pred - prev) ≠ sign(true - prev)).

    Args:
        params: Dict of model parameters (kappa, mu_scale, omega, alpha, beta).
        data_batch: Dict with keys u0, drift, close_target, log_returns, etc.
        model_fn: Callable (params, data_batch, noise_key) -> predicted_prices array.
        noise_key: JAX random key for noise generation.

    Returns:
        Scalar loss value.
    """
    predicted = model_fn(params, data_batch, noise_key)
    target = data_batch["close_target"]

    T = target.shape[0]
    n = jnp.minimum(T, predicted.shape[0])
    predicted = predicted[:n]
    target = target[:n]

    # MSE loss
    mse = jnp.mean((predicted - target) ** 2)

    # Directional penalty
    if n > 1:
        pred_diff = predicted[1:] - predicted[:-1]
        true_diff = target[1:] - target[:-1]
        # Fraction where signs disagree
        sign_mismatch = jnp.mean(
            (jnp.sign(pred_diff) != jnp.sign(true_diff)).astype(jnp.float32)
        )
    else:
        sign_mismatch = jnp.float32(0.0)

    # Flat-volatility penalty: approaches 1.0 when sigma is constant,
    # approaches 0.0 when sigma is dynamic — always bounded [0, 1]
    sigma = params.get("sigma_trajectory", None)
    flat_vol_penalty = (
        0.1 * jnp.exp(-100.0 * jnp.var(sigma))
        if sigma is not None
        else jnp.float32(0.0)
    )

    return mse + 0.1 * sign_mismatch + flat_vol_penalty
