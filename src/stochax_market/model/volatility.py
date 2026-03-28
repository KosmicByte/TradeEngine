"""GARCH(1,1) volatility field as an Equinox module."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class GARCHVolatility(eqx.Module):
    """GARCH(1,1) stochastic volatility model.

    σ²_t = ω + α X²_{t-1} + β σ²_{t-1}

    Parameters are stored in unconstrained space and transformed via bounded
    transforms to enforce meaningful ranges:
      omega = softplus(raw_omega)                         → ω > 0
      alpha = 0.05 + 0.15 * sigmoid(raw_alpha)           → α ∈ [0.05, 0.20]
      beta  = clip(0.50 + 0.40 * sigmoid(raw_beta),
                   max=0.999 - alpha)                     → β ∈ [0.50, 0.90]

    This guarantees alpha + beta < 1 (covariance-stationarity) while
    preventing degenerate near-zero ARCH/GARCH coefficients.

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
        raw_omega: float = -3.0,
        raw_alpha: float = 0.5,
        raw_beta: float = 2.0,
    ):
        """Initialize GARCH parameters directly in unconstrained space.

        Stores raw (unconstrained) values that are transformed at access time:
        omega = softplus(raw_omega),
        alpha = 0.05 + 0.15 * sigmoid(raw_alpha)  → alpha in [0.05, 0.20],
        beta  = 0.50 + 0.40 * sigmoid(raw_beta)   → beta  in [0.50, 0.90],
        with beta additionally clipped to keep alpha + beta < 1.

        The default initialisation (raw_omega=-3, raw_alpha=0.5, raw_beta=2)
        gives omega≈0.049, alpha≈0.12, beta≈0.85, which is a sensible
        GARCH(1,1) starting point with strong volatility persistence.

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
        return 0.05 + 0.15 * jax.nn.sigmoid(self.raw_alpha)

    @property
    def beta(self) -> jnp.ndarray:
        alpha = 0.05 + 0.15 * jax.nn.sigmoid(self.raw_alpha)
        beta_unconstrained = 0.50 + 0.40 * jax.nn.sigmoid(self.raw_beta)
        return jnp.minimum(beta_unconstrained, 0.999 - alpha)

    def __call__(self, log_returns: jnp.ndarray) -> jnp.ndarray:
        """Compute GARCH(1,1) conditional variance series.

        Args:
            log_returns: Shape (T,) array of log-returns.

        Returns:
            Shape (T,) array of conditional volatilities σ_t.
        """
        omega = self.omega
        alpha = self.alpha
        beta = self.beta

        sigma2_init = omega / (1.0 - alpha - beta + 1e-8)

        def scan_fn(carry, x_t):
            sigma2_prev = carry
            # Replace NaN returns (e.g. split-adjusted days) with 0 so the
            # ARCH term does not propagate NaN through the variance path.
            x_safe = jnp.where(jnp.isnan(x_t), 0.0, x_t)
            sigma2_t = omega + alpha * x_safe**2 + beta * sigma2_prev
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


