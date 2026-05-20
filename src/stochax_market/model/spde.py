"""Core SPDE stepper wrapping exponax for pseudo-spectral diffusion.

Advective formulation with learnable noise gain (v0.5+) and small diffusion (v1.0)
---------------------------------------------------------------------------------
The model is

    ∂u/∂t = κ∇²u − μ_scale·drift·∂u/∂x − σ_scale·σ(x)·∂u/∂x · Ẇ

Three scale parameters in front of the dynamics:

- ``mu_scale``     : drift amplification (learnable), advects u with drift
- ``sigma_scale``  : noise amplification (learnable), strength of stochastic
                     advection
- ``kappa``        : diffusion coefficient (FIXED, small), models intraday
                     uncertainty smearing

Why kappa is now small (v1.0, June 2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The original default ``kappa = 0.01`` was inherited from PDE-textbook
canonical values and ran for a year without anyone noticing the
consequences over long forecast horizons. With ``dt = 1/252``, the
Gaussian initial-condition width grows as σ²(t) = σ²(0) + 2κt, i.e.

    Δσ² per step = 2 · 0.01 / 252 ≈ 7.9e-5

Over 150 forecast steps that pushes σ from 0.05 to ≈ 0.12 in the
normalised [0, 1] domain. When the initial price is well off-center
(RELIANCE: x₀ = 1388/1611 ≈ 0.86), a Gaussian of width 0.12 has
significant mass at x > 1.0; with the exponax stepper's periodic
boundary conditions, this mass wraps around to x ≈ 0 and pollutes the
center-of-mass calculation in ``field_mean``. The visible symptom is
the "cliff dive" observed in the May 2026 RELIANCE backtest: the SPDE
simulation drifted from ₹1388 toward ₹950 over August-October — that
₹950 is just (1611 × 0.5) ≈ ₹805 plus stochastic-noise offset, where
0.5 is the domain midpoint that the wrap-around-confused
center-of-mass converges to.

Setting κ = 1e-4 (100× smaller) gives Δσ² per step ≈ 7.9e-7, so over
150 steps the Gaussian width grows from 0.05 to ≈ 0.051 — essentially
unchanged. Mass stays put. The model becomes a pure stochastic
advection process modulated by GARCH volatility, which is what we
actually want for a price-forecast SPDE.

If you ever need *more* diffusion (e.g. for intraday-resolution
modelling where intra-period uncertainty matters), bump kappa back up
but watch the wrap-around behaviour for prices near domain boundaries.
Making kappa learnable is the natural next step if a fixed value
proves too restrictive across instruments.

Boundary conditions
-------------------
Periodic, inherited from exponax. With κ small the periodic-BC
wrap-around is a non-issue for any realistic forecast horizon.

Ito vs Stratonovich
-------------------
Ito convention.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from exponax.stepper import Diffusion


def _spatial_gradient(u: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Central finite-difference ∂u/∂x with periodic boundary conditions.

    Uses ``jnp.roll`` for periodic neighbour access. Second-order accurate
    in the interior. Matches the periodic BC used by the exponax diffusion
    stepper so the two operators are consistent.
    """
    return (jnp.roll(u, -1) - jnp.roll(u, 1)) / (2.0 * dx)


class SPDEStepper(eqx.Module):
    """Stochastic PDE stepper for price field evolution (advective form).

    Implements:
        ∂u/∂t = κ∇²u − μ_scale·drift·∂u/∂x − σ_scale·σ(x)·∂u/∂x · Ẇ

    Attributes:
        diffusion_stepper: Pre-built exponax Diffusion stepper.
        raw_mu_scale: Unconstrained drift scaling parameter
            (``mu_scale = softplus(raw_mu_scale)``).
        raw_sigma_scale: Unconstrained noise scaling parameter
            (``sigma_scale = softplus(raw_sigma_scale)``).
        nx: Number of spatial grid points.
        dt: Time step size.
        domain_extent: Spatial domain extent L.
    """

    diffusion_stepper: Diffusion
    raw_mu_scale: jnp.ndarray
    raw_sigma_scale: jnp.ndarray
    nx: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    domain_extent: float = eqx.field(static=True)

    def __init__(
        self,
        kappa: float = 1e-4,
        mu_scale: float = 0.1,
        sigma_scale: float = 1.0,
        nx: int = 128,
        dt: float = 1.0 / 252.0,
        domain_extent: float = 1.0,
    ):
        """Initialize SPDE stepper.

        Args:
            kappa: Diffusion coefficient (concrete value for exponax).
                Default 1e-4 — see module docstring for the boundary-
                wrap-around rationale that motivated this value. The
                previous default (0.01) caused long-horizon forecasts
                to drift toward the spatial-domain midpoint due to
                periodic-BC wrap-around at extreme initial conditions.
            mu_scale: Drift scaling factor (positive; controls how strongly
                the historical drift signal advects u per step). Learnable.
            sigma_scale: Noise scaling factor (positive; multiplies the
                already-scaled σ(x) field into the stochastic-advection
                term). Learnable.
            nx: Number of spatial grid points.
            dt: Time step size.
            domain_extent: Spatial domain extent L.
        """
        self.diffusion_stepper = Diffusion(
            num_spatial_dims=1,
            domain_extent=domain_extent,
            num_points=nx,
            dt=dt,
            diffusivity=float(kappa),
        )
        self.raw_mu_scale    = _inverse_softplus(jnp.float32(mu_scale))
        self.raw_sigma_scale = _inverse_softplus(jnp.float32(sigma_scale))
        self.nx = nx
        self.dt = dt
        self.domain_extent = domain_extent

    @property
    def mu_scale(self) -> jnp.ndarray:
        return jax.nn.softplus(self.raw_mu_scale)

    @property
    def sigma_scale(self) -> jnp.ndarray:
        return jax.nn.softplus(self.raw_sigma_scale)

    def step(
        self,
        u: jnp.ndarray,
        sigma_field: jnp.ndarray,
        noise_increment: jnp.ndarray,
        drift_scalar: jnp.ndarray,
    ) -> jnp.ndarray:
        """Perform one SPDE timestep — advective form with learnable σ-gain.

        u_{n+1} = exponax_diff(u_n)
                  − μ_scale · drift · ∂u/∂x · Δt
                  − σ_scale · σ(x) · ∂u/∂x · ΔW
        """
        u_exp = u[None, :]
        u_diffused = self.diffusion_stepper(u_exp)[0]

        dx = self.domain_extent / float(self.nx)
        du_dx = _spatial_gradient(u, dx)

        drift_term = -self.mu_scale * drift_scalar * du_dx * self.dt
        noise_term = -self.sigma_scale * sigma_field * du_dx * noise_increment

        u_next = u_diffused + drift_term + noise_term
        u_next = jnp.maximum(u_next, 0.0)

        return u_next

    def rollout(
        self,
        u0: jnp.ndarray,
        sigma_trajectory: jnp.ndarray,
        noise_trajectory: jnp.ndarray,
        drift_series: jnp.ndarray,
        nt: int,
    ) -> jnp.ndarray:
        """Run full SPDE trajectory via jax.lax.scan.

        Signature is unchanged so fit.py / predict.py / simulate.py need no
        modifications.
        """

        def scan_fn(u, inputs):
            sigma_field, noise_inc, drift = inputs
            u_next = self.step(u, sigma_field, noise_inc, drift)
            return u_next, u_next

        inputs = (sigma_trajectory[:nt], noise_trajectory[:nt], drift_series[:nt])
        _, trajectory = jax.lax.scan(scan_fn, u0, inputs)

        return trajectory


def _inverse_softplus(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse of softplus: log(exp(x) - 1)."""
    return jnp.log(jnp.exp(x) - 1.0 + 1e-8)
