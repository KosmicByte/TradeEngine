"""Tests for SPDE stepper and model components."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from stochax_market.model.initial import price_to_field
from stochax_market.model.noise import make_noise_trajectory, make_wiener_sample
from stochax_market.model.spde import SPDEStepper
from stochax_market.model.volatility import GARCHVolatility


class TestSPDEStepper:
    """Tests for SPDEStepper."""

    def test_step_output_shape(self):
        """One SPDE step produces output with correct shape."""
        nx = 64
        spde = SPDEStepper(nx=nx)
        u = price_to_field(0.5, 1.0, nx)
        sigma_field = jnp.ones(nx, dtype=jnp.float32) * 0.02
        noise = jnp.zeros(nx, dtype=jnp.float32)
        drift = jnp.float32(0.01)

        u_next = spde.step(u, sigma_field, noise, drift)
        assert u_next.shape == (nx,)

    def test_step_preserves_nonnegativity(self):
        """SPDE step output is non-negative."""
        nx = 64
        spde = SPDEStepper(nx=nx)
        u = price_to_field(0.5, 1.0, nx)
        sigma_field = jnp.ones(nx, dtype=jnp.float32) * 0.02
        key = jax.random.key(0)
        noise = make_wiener_sample(
            key, nx, 8,
            jnp.arange(1, 9, dtype=jnp.float32) ** -2.0,
            1.0 / nx,
        ) * jnp.sqrt(1.0 / 252.0)
        drift = jnp.float32(0.01)

        u_next = spde.step(u, sigma_field, noise, drift)
        assert jnp.all(u_next >= 0.0)

    def test_rollout_output_shape(self):
        """Rollout produces trajectory with correct shape."""
        nx = 64
        nt = 5
        spde = SPDEStepper(nx=nx)
        u0 = price_to_field(0.5, 1.0, nx)
        sigma_traj = jnp.ones((nt, nx), dtype=jnp.float32) * 0.02
        noise_traj = jnp.zeros((nt, nx), dtype=jnp.float32)
        drift_series = jnp.zeros(nt, dtype=jnp.float32)

        trajectory = spde.rollout(u0, sigma_traj, noise_traj, drift_series, nt)
        assert trajectory.shape == (nt, nx)

    def test_rollout_noise_amplitude(self):
        """Noise term is additive and not suppressed: std of trajectory > 1.0."""
        nx = 64
        nt = 50
        spde = SPDEStepper(nx=nx)
        u0 = price_to_field(0.5, 1.0, nx)
        sigma_traj = jnp.ones((nt, nx), dtype=jnp.float32) * 0.5
        key = jax.random.key(0)
        noise_traj = make_noise_trajectory(key, nt, nx, spde.dt)
        drift_series = jnp.zeros(nt, dtype=jnp.float32)

        trajectory = spde.rollout(u0, sigma_traj, noise_traj, drift_series, nt)
        assert jnp.std(trajectory) > 1.0, (
            f"Expected std > 1.0 (noise not suppressed), got {jnp.std(trajectory):.4f}"
        )


        """SPDE step is differentiable w.r.t. mu_scale parameter."""
        import equinox as eqx

        nx = 64
        spde = SPDEStepper(nx=nx)
        u = price_to_field(0.5, 1.0, nx)
        sigma_field = jnp.ones(nx, dtype=jnp.float32) * 0.02
        noise = jnp.zeros(nx, dtype=jnp.float32)
        drift = jnp.float32(0.01)

        # Use equinox filter_grad to handle non-differentiable leaves
        @eqx.filter_grad
        def grad_fn(spde_model):
            u_next = spde_model.step(u, sigma_field, noise, drift)
            return jnp.sum(u_next)

        grad = grad_fn(spde)
        assert grad.raw_mu_scale is not None
        assert jnp.isfinite(grad.raw_mu_scale)


class TestGARCHVolatility:
    """Tests for GARCHVolatility."""

    def test_output_shape(self):
        """GARCH produces σ series with correct shape."""
        garch = GARCHVolatility()
        log_returns = jnp.array([0.01, -0.02, 0.005, -0.01, 0.03], dtype=jnp.float32)
        sigma = garch(log_returns)
        assert sigma.shape == (5,)

    def test_output_positive(self):
        """GARCH volatilities are positive."""
        garch = GARCHVolatility()
        log_returns = jnp.array([0.01, -0.02, 0.005, -0.01, 0.03], dtype=jnp.float32)
        sigma = garch(log_returns)
        assert jnp.all(sigma > 0.0)

    def test_spatial_field_shape(self):
        """to_spatial_field produces correct shape."""
        field = GARCHVolatility.to_spatial_field(jnp.float32(0.02), 128)
        assert field.shape == (128,)

    def test_stationarity_enforcement(self):
        """GARCH enforces finite σ even for large raw parameters."""
        garch = GARCHVolatility(raw_alpha=3.0, raw_beta=3.0)
        log_returns = jnp.array([0.01, -0.02, 0.005], dtype=jnp.float32)
        sigma = garch(log_returns)
        # Should not produce NaN or Inf
        assert jnp.all(jnp.isfinite(sigma))


class TestNoise:
    """Tests for noise generation."""

    def test_wiener_sample_shape(self):
        """Wiener sample has correct shape."""
        key = jax.random.key(0)
        nx = 64
        n_modes = 8
        eigenvalues = jnp.arange(1, n_modes + 1, dtype=jnp.float32) ** -2.0
        sample = make_wiener_sample(key, nx, n_modes, eigenvalues, 1.0 / nx)
        assert sample.shape == (nx,)

    def test_noise_trajectory_shape(self):
        """Noise trajectory has correct shape."""
        key = jax.random.key(1)
        noise = make_noise_trajectory(key, 10, 64, 1.0 / 252.0)
        assert noise.shape == (10, 64)

    def test_noise_different_keys(self):
        """Different keys produce different noise."""
        key1 = jax.random.key(0)
        key2 = jax.random.key(1)
        n1 = make_noise_trajectory(key1, 5, 32, 0.01)
        n2 = make_noise_trajectory(key2, 5, 32, 0.01)
        assert not jnp.allclose(n1, n2)


class TestInitialCondition:
    """Tests for initial condition generation."""

    def test_price_to_field_shape(self):
        """price_to_field produces correct shape."""
        u0 = price_to_field(0.5, 1.0, 128)
        assert u0.shape == (128,)

    def test_price_to_field_nonnegative(self):
        """Initial condition is non-negative."""
        u0 = price_to_field(0.5, 1.0, 128)
        assert jnp.all(u0 >= 0.0)

    def test_price_to_field_normalized(self):
        """Initial condition integrates approximately to 1."""
        nx = 128
        u0 = price_to_field(0.5, 1.0, nx)
        dx = 1.0 / (nx - 1)
        integral = jnp.sum(u0) * dx
        assert abs(float(integral) - 1.0) < 0.1
