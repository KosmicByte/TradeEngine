"""Q-Wiener process spatial noise via Karhunen-Loève expansion."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def make_wiener_sample(
    key: jax.Array,
    nx: int,
    n_modes: int,
    eigenvalues: jnp.ndarray,
    dx: float,
) -> jnp.ndarray:
    """Generate a single spatial Wiener sample via Karhunen-Loève expansion.

    W(x) = Σᵢ √λᵢ ξᵢ φᵢ(x)  where ξᵢ ~ N(0,1), φᵢ = √(2/L) sin(iπx/L)

    Args:
        key: JAX random key.
        nx: Number of spatial grid points.
        n_modes: Number of KL modes to use.
        eigenvalues: Shape (n_modes,) eigenvalues λᵢ of covariance operator Q.
        dx: Spatial grid spacing.

    Returns:
        Shape (nx,) spatial noise sample.
    """
    L = nx * dx
    x = jnp.linspace(0.0, L, nx, dtype=jnp.float32)

    xi = jax.random.normal(key, shape=(n_modes,), dtype=jnp.float32)

    modes = jnp.arange(1, n_modes + 1, dtype=jnp.float32)
    # φᵢ(x) = √(2/L) sin(iπx/L), shape: (n_modes, nx)
    phi = jnp.sqrt(2.0 / L) * jnp.sin(
        modes[:, None] * jnp.pi * x[None, :] / L
    )

    # W(x) = Σ √λᵢ ξᵢ φᵢ(x)
    sqrt_lam = jnp.sqrt(eigenvalues[:n_modes])
    sample = jnp.sum(sqrt_lam[:, None] * xi[:, None] * phi, axis=0)

    return sample


def make_noise_trajectory(
    key: jax.Array,
    nt: int,
    nx: int,
    dt: float,
    n_modes: int = 32,
    decay_rate: float = 2.0,
    dx: float | None = None,
    empirical_sigma: float = 0.015,
) -> jnp.ndarray:
    """Generate a noise trajectory of shape (nt, nx) with √dt scaling.

    Threads the random key through jax.lax.scan so each timestep uses a
    distinct subkey derived from the previous one, guaranteeing independent
    noise increments per step.

    Eigenvalues are set to λᵢ = (empirical_sigma / i)^decay_rate, directly
    anchored to the observed daily log-return std. The field produced by
    make_initial_condition has unit integral (∫u dx = 1), so the noise
    increment per step must be O(empirical_sigma) to produce price moves
    commensurate with observed daily volatility. The field_scale correction
    previously applied cancelled out empirical_sigma entirely, reducing
    eigenvalues to (1/nx/i)² — grid-spacing scale, not volatility scale.

    Args:
        key: JAX random key (first positional argument).
        nt: Number of timesteps.
        nx: Number of spatial grid points.
        dt: Time step size.
        n_modes: Number of Karhunen-Loève modes (default 32).
        decay_rate: Power-law decay exponent for eigenvalues: λᵢ ∝ i^{-decay_rate}.
        dx: Spatial grid spacing. Defaults to 1.0/nx.
        empirical_sigma: Observed daily log-return std used to anchor noise
            amplitude (default 0.015, ≈ RELIANCE daily vol). Eigenvalues are
            λᵢ = (empirical_sigma / i)^decay_rate, so mode 1 has amplitude
            empirical_sigma = 0.015 and higher modes decay as 1/i².

    Returns:
        Shape (nt, nx) noise increments √dt · W_sample per timestep.
    """
    if dx is None:
        dx = 1.0 / nx

    modes = jnp.arange(1, n_modes + 1, dtype=jnp.float32)

    # λᵢ = (empirical_sigma / i)^decay_rate
    # Mode 1 amplitude: empirical_sigma = 0.015 (≈ daily log-return std)
    # Mode i amplitude: empirical_sigma / i^(decay_rate/2) after sqrt
    eigenvalues = (empirical_sigma / modes) ** decay_rate

    def _step(carry_key: jax.Array, _: None) -> tuple[jax.Array, jnp.ndarray]:
        carry_key, subkey = jax.random.split(carry_key)
        sample = make_wiener_sample(subkey, nx, n_modes, eigenvalues, dx)
        return carry_key, sample

    _, samples = jax.lax.scan(_step, key, None, length=nt)  # (nt, nx)

    return jnp.sqrt(dt) * samples