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
        omega: float = 0.1,
        alpha: float = 0.1,
        beta: float = 0.8,
    ):
        """Initialize GARCH parameters.

        Defaults (ω=0.1, α=0.1, β=0.8) satisfy the stationarity condition
        α + β = 0.9 < 1 and produce a non-trivial long-run variance of
        ω / (1 − α − β) = 1.0, ensuring physically-meaningful volatility
        before any calibration step.

        Args:
            omega: Base variance (ω > 0).
            alpha: ARCH coefficient (α ≥ 0).
            beta: GARCH coefficient (β ≥ 0).
        """
        self.raw_omega = _inverse_softplus(jnp.float32(omega))
        self.raw_alpha = _inverse_softplus(jnp.float32(alpha))
        self.raw_beta = _inverse_softplus(jnp.float32(beta))

    @property
    def omega(self) -> jnp.ndarray:
        return jax.nn.softplus(self.raw_omega)

    @property
    def alpha(self) -> jnp.ndarray:
        return jax.nn.softplus(self.raw_alpha)

    @property
    def beta(self) -> jnp.ndarray:
        return jax.nn.softplus(self.raw_beta)

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

        Uses a constant base plus small sinusoidal perturbation for spatial
        structure.

        Args:
            sigma_t: Scalar volatility value.
            nx: Number of spatial grid points.

        Returns:
            Shape (nx,) spatial volatility field.
        """
        x = jnp.linspace(0.0, 1.0, nx, dtype=jnp.float32)
        perturbation = 0.05 * jnp.sin(2.0 * jnp.pi * x)
        return sigma_t * (1.0 + perturbation)


def _inverse_softplus(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse of softplus: log(exp(x) - 1)."""
    return jnp.log(jnp.exp(x) - 1.0 + 1e-8)
