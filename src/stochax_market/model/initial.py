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
        price: Center of the Gaussian (normalized price coordinate in [0, L]).
        L: Length of the spatial domain [0, L].
        nx: Number of spatial grid points.
        width: Standard deviation of the Gaussian blob (same units as price).

    Returns:
        Shape (nx,) normalized Gaussian initial condition.
    """
    x = jnp.linspace(0.0, L, nx, dtype=jnp.float32)
    u0 = jnp.exp(-((x - price) ** 2) / (2.0 * width**2))
    dx = L / max(nx - 1, 1)
    integral = jnp.sum(u0) * dx
    u0 = u0 / (integral + 1e-8)
    return u0


def make_initial_condition(
    last_price: float,
    domain_extent: float,
    nx: int,
    sigma_width: float = 50.0,
) -> jnp.ndarray:
    """Create a normalised Gaussian initial condition from a raw price.

    Converts a raw price (e.g. ₹2000) to the normalised grid coordinate
    ``x0 = last_price / domain_extent`` and calls :func:`price_to_field`.
    The Gaussian width is also normalised so it stays proportional to the
    domain regardless of the absolute price level.

    Args:
        last_price: Raw closing price (e.g. 2000.0 INR).
        domain_extent: Spatial domain length L (e.g. 3298.0, the max High
            across the full dataset).  If ``last_price > domain_extent``
            the normalised coordinate is clipped to 1.0 so the Gaussian
            stays on the grid.
        nx: Number of spatial grid points.
        sigma_width: Gaussian half-width in the same units as ``last_price``
            (default 50.0 INR, normalised to ≈ 0.015 on the unit grid).

    Returns:
        Shape (nx,) normalized Gaussian initial condition on [0, 1].
    """
    x0 = last_price / domain_extent            # normalised coordinate
    x0 = float(jnp.clip(jnp.float32(x0), 0.0, 1.0))  # keep on the [0,1] grid
    width_norm = sigma_width / domain_extent    # normalised width
    return price_to_field(x0, L=1.0, nx=nx, width=width_norm)
