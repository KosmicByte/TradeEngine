"""Map Prev Close to Gaussian initial condition u₀(x)."""

from __future__ import annotations

import jax.numpy as jnp


def price_to_field(
    price: float,
    L: float,
    nx: int,
    width: float = 0.05,
) -> jnp.ndarray:
    """Create a Gaussian blob initial condition centered at the given price.

    u₀(x) = exp(-(x - price)² / (2·width²)), normalized to integrate to 1.

    Args:
        price: Center of the Gaussian (normalized price coordinate).
        L: Length of the spatial domain [0, L].
        nx: Number of spatial grid points.
        width: Standard deviation of the Gaussian blob.

    Returns:
        Shape (nx,) normalized Gaussian initial condition.
    """
    x = jnp.linspace(0.0, L, nx, dtype=jnp.float32)
    u0 = jnp.exp(-((x - price) ** 2) / (2.0 * width**2))
    dx = L / (nx - 1 + 1e-8)
    integral = jnp.sum(u0) * dx
    u0 = u0 / (integral + 1e-8)
    return u0
