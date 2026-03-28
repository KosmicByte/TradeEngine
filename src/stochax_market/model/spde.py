"""Core SPDE stepper wrapping exponax for pseudo-spectral diffusion."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from exponax.stepper import Diffusion


class SPDEStepper(eqx.Module):
    """Stochastic PDE stepper for price field evolution.

    Implements: ∂u/∂t = κ∇²u + μ_scale·drift·u + σ(x)·u·Ẇ

    The linear diffusion κ∇²u is handled by exponax's pseudo-spectral
    exponential time differencing. The multiplicative noise and drift terms
    are applied in physical space after each diffusion step.

    The exponax diffusion stepper is pre-created at initialization with
    concrete κ and stored as a module attribute. The differentiable
    parameter mu_scale controls the drift scaling.

    Attributes:
        diffusion_stepper: Pre-built exponax Diffusion stepper.
        raw_mu_scale: Unconstrained drift scaling parameter.
        nx: Number of spatial grid points.
        dt: Time step size.
        domain_extent: Spatial domain extent L.
    """

    diffusion_stepper: Diffusion
    raw_mu_scale: jnp.ndarray
    nx: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    domain_extent: float = eqx.field(static=True)

    def __init__(
        self,
        kappa: float = 0.01,
        mu_scale: float = 0.1,
        nx: int = 128,
        dt: float = 1.0 / 252.0,
        domain_extent: float = 1.0,
    ):
        """Initialize SPDE stepper.

        Args:
            kappa: Diffusion coefficient (concrete value for exponax).
            mu_scale: Drift scaling factor.
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
        self.raw_mu_scale = _inverse_softplus(jnp.float32(mu_scale))
        self.nx = nx
        self.dt = dt
        self.domain_extent = domain_extent

    @property
    def mu_scale(self) -> jnp.ndarray:
        return jax.nn.softplus(self.raw_mu_scale)

    def step(
        self,
        u: jnp.ndarray,
        sigma_field: jnp.ndarray,
        noise_increment: jnp.ndarray,
        drift_scalar: jnp.ndarray,
    ) -> jnp.ndarray:
        """Perform one SPDE timestep.

        u_{n+1} = exponax_step(u_n) + σ(x) * u_n * ΔW + μ_scale * drift * u_n * Δt

        Args:
            u: Shape (nx,) current field state.
            sigma_field: Shape (nx,) spatial volatility field.
            noise_increment: Shape (nx,) noise increment √dt·W.
            drift_scalar: Scalar drift value for this timestep.

        Returns:
            Shape (nx,) updated field state.
        """
        # exponax expects (1, nx) shape; add and remove channel dim
        u_exp = u[None, :]
        u_diffused = self.diffusion_stepper(u_exp)[0]

        # Multiplicative noise: σ(x) * u * ΔW
        noise_term = sigma_field * u * noise_increment

        # Drift term: μ_scale * drift * u * Δt
        drift_term = self.mu_scale * drift_scalar * u * self.dt

        u_next = u_diffused + noise_term + drift_term

        # Ensure non-negativity (probability-like field)
        u_next = jnp.maximum(u_next, 0.0)

        # Renormalize to prevent blow-up
        u_next = u_next / (jnp.sum(u_next) * self.domain_extent / self.nx + 1e-8)

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

        Args:
            u0: Shape (nx,) initial condition.
            sigma_trajectory: Shape (nt, nx) volatility fields per timestep.
            noise_trajectory: Shape (nt, nx) noise increments per timestep.
            drift_series: Shape (nt,) drift values per timestep.
            nt: Number of timesteps to simulate.

        Returns:
            Shape (nt, nx) trajectory of field states.
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
