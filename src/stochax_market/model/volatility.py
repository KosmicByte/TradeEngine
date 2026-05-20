"""GARCH(1,1) volatility field as an Equinox module."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


# Persistence cap. Constrains α + β ≤ this value via the β-property's
# jnp.minimum clip. Empirical equity GARCH(1,1) fits typically have
# persistence in [0.94, 0.99] with vol half-lives of 12–70 days. Setting
# the cap at 0.97 (half-life ≈ 23 days) gives BFGS enough room to find
# realistic persistence values without allowing the unit-root corner
# that v0.6/v0.7 exposed: when the rollout shift-variance penalty in
# fit.py creates pressure toward bursty volatility, BFGS would drive
# persistence to the stationarity boundary (0.999) and shrink ω to keep
# unconditional variance anchored — producing models with 693-day vol
# half-lives and runaway long-horizon forecasts (RELIANCE v0.7 backtest:
# α at 0.20, β at the 0.999-α boundary, α+β = 0.999).
#
# 0.97 is the empirically-reasonable upper bound for daily equity GARCH;
# values much higher than this are pathological for this model not because
# of GARCH theory (which permits up to 1) but because the SPDE rollout
# integrates GARCH σ_t multiplicatively over many days, and near-unit-root
# σ_t trajectories produce unrealistic burst-then-stay-high behaviour.
_PERSISTENCE_CAP: float = 0.97


class GARCHVolatility(eqx.Module):
    """GARCH(1,1) stochastic volatility model.

    σ²_t = ω + α X²_{t-1} + β σ²_{t-1}

    Parameters are stored in unconstrained space and transformed via bounded
    transforms to enforce meaningful ranges:
      omega = softplus(raw_omega)                         → ω > 0
      alpha = 0.05 + 0.15 * sigmoid(raw_alpha)           → α ∈ [0.05, 0.20]
      beta  = clip(0.50 + 0.40 * sigmoid(raw_beta),
                   max = _PERSISTENCE_CAP - alpha)        → β ∈ [0.50, 0.90]

    This guarantees α + β ≤ _PERSISTENCE_CAP (0.97) — well below the
    stationarity boundary of 1, but tight enough to prevent the
    unit-root pathologies exposed by the v0.6/v0.7 RELIANCE fits where
    BFGS pushed α+β → 0.999 to exploit the rollout shift-variance penalty.

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
        raw_omega: float = -11.0,
        raw_alpha: float = 0.5,
        raw_beta: float = 2.0,
    ):
        """Initialize GARCH parameters directly in unconstrained space.

        Stores raw (unconstrained) values that are transformed at access time:
        omega = softplus(raw_omega),
        alpha = 0.05 + 0.15 * sigmoid(raw_alpha)  → alpha in [0.05, 0.20],
        beta  = 0.50 + 0.40 * sigmoid(raw_beta)   → beta  in [0.50, 0.90],
        with beta additionally clipped to keep α + β ≤ _PERSISTENCE_CAP.

        The default initialisation (raw_omega=-11, raw_alpha=0.5, raw_beta=2)
        gives:
            omega ≈ 1.67e-5,  alpha ≈ 0.125,  beta ≈ clipped to 0.845
            persistence       α + β = 0.97 (at the cap; BFGS will explore
                              the [0.55, 0.97] range)
            unconditional σ²  ≈ ω / (1 − α − β) → set by ω given persistence

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
        return jnp.minimum(beta_unconstrained, _PERSISTENCE_CAP - alpha)

    def __call__(self, log_returns: jnp.ndarray) -> jnp.ndarray:
        """Compute GARCH(1,1) conditional volatility series.

        Args:
            log_returns: Shape (T,) array of log-returns.

        Returns:
            Shape (T,) array of conditional volatilities σ_t (daily, log-return scale).
        """
        omega = self.omega
        alpha = self.alpha
        beta  = self.beta

        sigma2_init = omega / (1.0 - alpha - beta + 1e-8)

        def scan_fn(carry, x_t):
            sigma2_prev = carry
            # Replace NaN returns (e.g. split-adjusted days) with 0 so the
            # ARCH term does not propagate NaN through the variance path.
            x_safe  = jnp.where(jnp.isnan(x_t), 0.0, x_t)
            sigma2_t = omega + alpha * x_safe ** 2 + beta * sigma2_prev
            sigma2_t = jnp.maximum(sigma2_t, 1e-8)
            return sigma2_t, sigma2_t

        _, sigma2_series = jax.lax.scan(scan_fn, sigma2_init, log_returns)

        return jnp.sqrt(sigma2_series)

    @staticmethod
    def to_spatial_field(sigma_t: jnp.ndarray, nx: int) -> jnp.ndarray:
        """Broadcast scalar σ_t to a spatial field of shape (nx,).

        Uses a half-period sinusoidal modulation for non-trivial spatial
        structure with guaranteed non-zero amplitude. The base nx multiplier
        is preserved from the multiplicative-noise era; the advective SPDE
        in spde.py (v0.4+) compensates via a learnable ``sigma_scale``
        parameter so this scalar's exact value is not critical.

        Args:
            sigma_t: Scalar volatility value (daily log-return scale).
            nx: Number of spatial grid points.

        Returns:
            Shape (nx,) spatial volatility field.
        """
        return sigma_t * nx * (
            1.0 + 0.1 * jnp.sin(jnp.linspace(0, jnp.pi, nx, dtype=jnp.float32))
        )
