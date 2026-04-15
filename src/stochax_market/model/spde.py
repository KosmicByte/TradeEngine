"""Core SPDE stepper wrapping exponax for pseudo-spectral diffusion."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from exponax.stepper import Diffusion


def _field_mean(u: jnp.ndarray, x_grid: jnp.ndarray) -> jnp.ndarray:
    """Compute field mean ∫ x·u(x) dx = Σ x_j·u_j·Δx."""
    dx = x_grid[1] - x_grid[0]
    return jnp.sum(x_grid * u) * dx


class PIDController(eqx.Module):
    """Differentiable field-space PID controller for SPDE drift correction.

    At each timestep t the controller computes:

        e_t  = r_t - ū_t,   ū_t = ∫ x·u(x,t) dx  (field mean)
        integral_t = clip(integral_{t-1} + e_t·Δt, -i_max, i_max)
        deriv_t    = (e_t - e_{t-1}) / Δt
        u_PID(t)   = Kp·e_t + Ki·integral_t + Kd·deriv_t

    and returns u_PID as an additive correction to the spatially-uniform drift:

        μ_corrected = μ_scale·drift_scalar + u_PID(t)

    Parameters are stored in unconstrained space and transformed via
    ``softplus`` to guarantee positivity:

        Kp = softplus(raw_Kp) > 0
        Ki = softplus(raw_Ki) > 0
        Kd = softplus(raw_Kd) > 0

    The controller state ``(integral, prev_error)`` is **not** stored as
    module state; it is threaded as a scan carry through ``jax.lax.scan``
    so that the rollout remains fully differentiable.

    Attributes:
        raw_Kp: Unconstrained proportional gain parameter.
        raw_Ki: Unconstrained integral gain parameter.
        raw_Kd: Unconstrained derivative gain parameter.
        i_max: Anti-windup clipping bound for the integral carry.
    """

    raw_Kp: jnp.ndarray
    raw_Ki: jnp.ndarray
    raw_Kd: jnp.ndarray
    i_max: float = eqx.field(static=True)

    def __init__(
        self,
        Kp: float = 0.1,
        Ki: float = 0.01,
        Kd: float = 0.001,
        i_max: float = 1.0,
    ):
        """Initialise PID gains in constrained space.

        Args:
            Kp: Proportional gain (must be > 0).
            Ki: Integral gain (must be > 0).
            Kd: Derivative gain (must be > 0).
            i_max: Anti-windup saturation bound for the integral carry.
        """
        self.raw_Kp = _inverse_softplus(jnp.float32(Kp))
        self.raw_Ki = _inverse_softplus(jnp.float32(Ki))
        self.raw_Kd = _inverse_softplus(jnp.float32(Kd))
        self.i_max = i_max

    @property
    def Kp(self) -> jnp.ndarray:
        return jax.nn.softplus(self.raw_Kp)

    @property
    def Ki(self) -> jnp.ndarray:
        return jax.nn.softplus(self.raw_Ki)

    @property
    def Kd(self) -> jnp.ndarray:
        return jax.nn.softplus(self.raw_Kd)

    def __call__(
        self,
        carry: tuple[jnp.ndarray, jnp.ndarray],
        u_mean: jnp.ndarray,
        setpoint: jnp.ndarray,
        dt: float,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """Compute PID correction for one timestep.

        Args:
            carry: ``(integral, prev_error)`` from the previous step.
                Both are scalar ``jnp.float32`` values.
            u_mean: Current field mean ū_t (normalised price coordinate).
            setpoint: Target value r_t (e.g. VWAP, normalised).
            dt: Time step size Δt.

        Returns:
            Tuple ``(u_pid, new_carry)`` where

            - ``u_pid`` — scalar additive drift correction.
            - ``new_carry`` — updated ``(integral, prev_error)`` tuple.
        """
        integral, prev_error = carry
        error = setpoint - u_mean
        integral = jnp.clip(integral + error * dt, -self.i_max, self.i_max)
        deriv = (error - prev_error) / dt
        u_pid = self.Kp * error + self.Ki * integral + self.Kd * deriv
        return u_pid, (integral, error)


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
        u_pid_correction: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Perform one SPDE timestep.

        u_{n+1} = exponax_step(u_n) + σ(x) * u_n * ΔW
                  + (μ_scale * drift + u_pid) * u_n * Δt

        Args:
            u: Shape (nx,) current field state.
            sigma_field: Shape (nx,) spatial volatility field.
            noise_increment: Shape (nx,) noise increment √dt·W.
            drift_scalar: Scalar drift value for this timestep.
            u_pid_correction: Optional scalar PID output to add to the drift.
                Defaults to zero when not provided.

        Returns:
            Shape (nx,) updated field state.
        """
        # exponax expects (1, nx) shape; add and remove channel dim
        u_exp = u[None, :]
        u_diffused = self.diffusion_stepper(u_exp)[0]

        # Multiplicative noise: σ(x) * u * ΔW
        noise_term = sigma_field * u * noise_increment

        # Drift term: (μ_scale * drift + u_pid) * u * Δt
        pid_correction = (
            jnp.float32(0.0) if u_pid_correction is None else u_pid_correction
        )
        drift_term = (self.mu_scale * drift_scalar + pid_correction) * u * self.dt

        u_next = u_diffused + noise_term + drift_term

        # Soft floor: prevent negative probability density without
        # renormalising — renormalisation would destroy price information
        # by forcing field_mean = ∫x·u dx to a constant every step.
        u_next = jnp.maximum(u_next, 0.0)

        return u_next

    def rollout(
        self,
        u0: jnp.ndarray,
        sigma_trajectory: jnp.ndarray,
        noise_trajectory: jnp.ndarray,
        drift_series: jnp.ndarray,
        nt: int,
        pid: PIDController | None = None,
        setpoint_series: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Run full SPDE trajectory via jax.lax.scan.

        When *pid* and *setpoint_series* are supplied the scan carry is
        extended to ``(u, integral, prev_error)`` and the PID correction is
        computed and injected into the drift at every step.  When omitted
        the original ``u``-only carry is used, preserving backward
        compatibility.

        Args:
            u0: Shape (nx,) initial condition.
            sigma_trajectory: Shape (nt, nx) volatility fields per timestep.
            noise_trajectory: Shape (nt, nx) noise increments per timestep.
            drift_series: Shape (nt,) drift values per timestep.
            nt: Number of timesteps to simulate.
            pid: Optional :class:`PIDController`.  If ``None`` no PID
                correction is applied.
            setpoint_series: Shape (nt,) target values r_t (normalised).
                Required when *pid* is not ``None``.

        Returns:
            Shape (nt, nx) trajectory of field states.
        """
        if pid is not None and setpoint_series is not None:
            x_grid = jnp.linspace(
                0.0, self.domain_extent, self.nx, dtype=jnp.float32
            )

            def scan_fn_pid(carry, inputs):
                u, integral, prev_error = carry
                sigma_field, noise_inc, drift, setpoint = inputs
                u_mean = _field_mean(u, x_grid)
                u_pid, (integral, prev_error) = pid(
                    (integral, prev_error), u_mean, setpoint, self.dt
                )
                u_next = self.step(u, sigma_field, noise_inc, drift, u_pid)
                return (u_next, integral, prev_error), u_next

            carry0 = (u0, jnp.float32(0.0), jnp.float32(0.0))
            inputs = (
                sigma_trajectory[:nt],
                noise_trajectory[:nt],
                drift_series[:nt],
                setpoint_series[:nt],
            )
            _, trajectory = jax.lax.scan(scan_fn_pid, carry0, inputs)
        else:
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