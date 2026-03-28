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

    Eigenvalues are scaled to the normalised field magnitude (1/nx) so that
    per-step noise increments are commensurate with the field values produced
    by make_initial_condition. Without this scaling, noise increments are
    O(empirical_sigma²) ≈ 2e-4, far below the field magnitude O(1/nx) ≈ 8e-3,
    making the stochastic term invisible.

    Args:
        key: JAX random key (first positional argument).
        nt: Number of timesteps.
        nx: Number of spatial grid points.
        dt: Time step size.
        n_modes: Number of Karhunen-Loève modes (default 32).
        decay_rate: Power-law decay exponent for eigenvalues: λᵢ ∝ i^{-decay_rate}.
        dx: Spatial grid spacing. Defaults to 1.0/nx.
        empirical_sigma: Observed daily log-return standard deviation used to
            anchor the noise amplitude (default 0.015, ≈ RELIANCE daily vol).
            Eigenvalues are scaled to field magnitude (1/nx) so that the noise
            term is commensurate with actual field values, then modulated by
            empirical_sigma / i^decay_rate to preserve the frequency decay.

    Returns:
        Shape (nt, nx) noise increments √dt · W_sample per timestep.
    """
    if dx is None:
        dx = 1.0 / nx

    modes = jnp.arange(1, n_modes + 1, dtype=jnp.float32)

    # Scale eigenvalues to normalised field magnitude (1/nx) so noise
    # increments are commensurate with field values from make_initial_condition.
    # Factor (1/nx) / empirical_sigma converts from log-return scale to field scale.
    field_scale = (1.0 / nx) / empirical_sigma
    eigenvalues = (field_scale * empirical_sigma / modes) ** decay_rate

    def _step(carry_key: jax.Array, _: None) -> tuple[jax.Array, jnp.ndarray]:
        carry_key, subkey = jax.random.split(carry_key)
        sample = make_wiener_sample(subkey, nx, n_modes, eigenvalues, dx)
        return carry_key, sample

    _, samples = jax.lax.scan(_step, key, None, length=nt)  # (nt, nx)

    return jnp.sqrt(dt) * samples