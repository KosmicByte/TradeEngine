"""Loss functions for calibration."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def field_mean(u: jnp.ndarray, x_grid: jnp.ndarray) -> jnp.ndarray:
    """Compute the expected price from a probability-like field.

    Center-of-mass formulation:
        ⟨x⟩ = (∫ x·u dx) / (∫ u dx + ε)

    The (∫ u dx) denominator is the explicit mass normaliser. The earlier
    formulation assumed mass conservation (∫ u dx = 1) and used the bare
    numerator, but the SPDE rollout can leak or accumulate mass when the
    soft floor `max(u, 0)` clips negative excursions or when the diffusion
    + multiplicative-noise scheme is not strictly mass-preserving. Dividing
    by the actual mass keeps the extracted price tracking the field's peak
    even when mass drifts during the rollout, which was a key source of
    bias before this fix.

    Args:
        u: Shape (nx,) field on the spatial grid.
        x_grid: Shape (nx,) spatial grid coordinates.

    Returns:
        Scalar expected value (first moment of the normalised field).
    """
    dx = x_grid[1] - x_grid[0]
    mass = jnp.sum(u) * dx
    return jnp.sum(x_grid * u) * dx / (mass + 1e-8)


def calibration_loss(
    params: dict,
    data_batch: dict[str, jnp.ndarray],
    model_fn: callable,
    noise_key: jax.Array,
) -> jnp.ndarray:
    """Compute calibration loss: MSE + directional + flat-vol penalty.

    Reference implementation of the penalised objective. The same loss is
    inlined inside `fit.py::_penalised_loss` against the Equinox-module API
    used by BFGS; this function is the dict/model_fn-based form used in
    tests and notebooks.

    Components
    ----------
    - MSE between predicted prices and ``close_target``.
    - 0.1 × directional penalty: fraction of timesteps where
      sign(pred_t − pred_{t-1}) ≠ sign(true_t − true_{t-1}).
    - Flat-volatility penalty: 0.1 · exp(−100 · var(σ_trajectory)) — pushes
      the optimiser away from degenerate solutions where σ collapses to a
      constant (the regression that produced std-ratio ≈ 0.148 in the
      RELIANCE backtest).

    Args:
        params: Dict of model parameters; optionally contains
            'sigma_trajectory' for the flat-vol penalty.
        data_batch: Dict with keys u0, drift, close_target, log_returns, etc.
        model_fn: Callable (params, data_batch, noise_key) → predicted_prices.
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
        sign_mismatch = jnp.mean(
            (jnp.sign(pred_diff) != jnp.sign(true_diff)).astype(jnp.float32)
        )
    else:
        sign_mismatch = jnp.float32(0.0)

    # Flat-volatility penalty: → 0.1 when sigma is constant, → 0 when dynamic.
    sigma = params.get("sigma_trajectory", None)
    flat_vol_penalty = (
        0.1 * jnp.exp(-100.0 * jnp.var(sigma))
        if sigma is not None
        else jnp.float32(0.0)
    )

    return mse + 0.1 * sign_mismatch + flat_vol_penalty
