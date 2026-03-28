"""GARCH(1,1) volatility field as an Equinox module."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class GARCHVolatility(eqx.Module):
    """GARCH(1,1) stochastic volatility model.

    σ²_t = ω + α X²_{t-1} + β σ²_{t-1}

    Parameters are stored in unconstrained space and transformed via softplus
    to enforce positivity. The stationarity condition α + β < 1 is enforced
    via a sigmoid rescaling.

    Attributes:
        raw_omega: Unconstrained parameter for ω.
        raw_alpha: Unconstrained parameter for α.
        raw_beta: Unconstrained parameter for β.
    """

    raw_omega: jnp.ndarray
    raw_alpha: jnp.ndarray
    raw_beta: jnp.ndarray

    def __init__(
        self,
        raw_omega: float = 0.5,
        raw_alpha: float = 0.3,
        raw_beta: float = 1.5,
    ):
        """Initialize GARCH parameters directly in unconstrained space.

        Stores raw (unconstrained) values that are transformed at access time:
        omega = softplus(raw_omega), alpha = softplus(raw_alpha),
        beta = softplus(raw_beta) * (1 - alpha - 1e-3). The stationarity
        condition alpha + beta < 1 is additionally enforced inside __call__
        as a safety net.

        Args:
            raw_omega: Unconstrained parameter for ω (stored directly).
            raw_alpha: Unconstrained parameter for α (stored directly).
            raw_beta: Unconstrained parameter for β (stored directly).
        """
        self.raw_omega = jnp.float32(raw_omega)
        self.raw_alpha = jnp.float32(raw_alpha)
        self.raw_beta = jnp.float32(raw_beta)

    @property
    def omega(self) -> jnp.ndarray:
        return jax.nn.softplus(self.raw_omega)

    @property
    def alpha(self) -> jnp.ndarray:
        return jax.nn.softplus(self.raw_alpha)

    @property
    def beta(self) -> jnp.ndarray:
        return jax.nn.softplus(self.raw_beta) * (
            1.0 - jax.nn.softplus(self.raw_alpha) - 1e-3
        )

    def __call__(self, log_returns: jnp.ndarray) -> jnp.ndarray:
        """Compute GARCH(1,1) conditional variance series.

        Args:
            log_returns: Shape (T,) array of log-returns.

        Returns:
            Shape (T,) array of conditional volatilities σ_t.
        """
        omega = self.omega
        alpha_raw = self.alpha
        beta_raw = self.beta

        # Enforce stationarity: rescale so alpha + beta < 1
        total = alpha_raw + beta_raw
        scale = jnp.where(total >= 0.99, 0.98 / (total + 1e-8), 1.0)
        alpha = alpha_raw * scale
        beta = beta_raw * scale

        sigma2_init = omega / (1.0 - alpha - beta + 1e-8)

        def scan_fn(carry, x_t):
            sigma2_prev = carry
            sigma2_t = omega + alpha * x_t**2 + beta * sigma2_prev
            sigma2_t = jnp.maximum(sigma2_t, 1e-8)
            return sigma2_t, sigma2_t

        _, sigma2_series = jax.lax.scan(scan_fn, sigma2_init, log_returns)

        return jnp.sqrt(sigma2_series)

    @staticmethod
    def to_spatial_field(sigma_t: jnp.ndarray, nx: int) -> jnp.ndarray:
        """Broadcast scalar σ_t to a spatial field of shape (nx,).

        Uses a half-period sinusoidal modulation for non-trivial spatial
        structure with guaranteed non-zero amplitude.

        Args:
            sigma_t: Scalar volatility value.
            nx: Number of spatial grid points.

        Returns:
            Shape (nx,) spatial volatility field.
        """
        return sigma_t * (
            1.0 + 0.1 * jnp.sin(jnp.linspace(0, jnp.pi, nx, dtype=jnp.float32))
        )


