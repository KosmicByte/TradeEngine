"""Tests for calibration: loss functions and fitting."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from stochax_market.calibration.loss import calibration_loss, field_mean
from stochax_market.model.initial import price_to_field
from stochax_market.model.spde import SPDEStepper
from stochax_market.model.volatility import GARCHVolatility


class TestFieldMean:
    """Tests for field_mean."""

    def test_field_mean_centered(self):
        """field_mean returns approximate center for symmetric distribution."""
        nx = 128
        x_grid = jnp.linspace(0.0, 1.0, nx, dtype=jnp.float32)
        u = price_to_field(0.5, 1.0, nx)
        mean = field_mean(u, x_grid)
        assert abs(float(mean) - 0.5) < 0.05

    def test_field_mean_differentiable(self):
        """field_mean is differentiable via jax.grad."""
        nx = 64
        x_grid = jnp.linspace(0.0, 1.0, nx, dtype=jnp.float32)

        def loss(u):
            return field_mean(u, x_grid)

        u = price_to_field(0.5, 1.0, nx)
        grad = jax.grad(loss)(u)
        assert grad.shape == (nx,)
        assert jnp.all(jnp.isfinite(grad))


class TestCalibrationLoss:
    """Tests for calibration_loss."""

    def test_loss_is_scalar(self):
        """calibration_loss returns a scalar."""
        nx = 64
        T = 5

        def dummy_model_fn(params, data, key):
            return jnp.ones(T, dtype=jnp.float32) * 0.5

        data_batch = {
            "u0": jnp.ones((T, nx), dtype=jnp.float32),
            "drift": jnp.zeros(T, dtype=jnp.float32),
            "close_target": jnp.ones(T, dtype=jnp.float32) * 0.5,
            "log_returns": jnp.zeros(T, dtype=jnp.float32),
        }

        loss = calibration_loss(
            {}, data_batch, dummy_model_fn, jax.random.key(0)
        )
        assert loss.shape == ()

    def test_loss_zero_for_perfect_prediction(self):
        """Loss is zero when prediction matches target exactly."""
        T = 10

        def perfect_model(params, data, key):
            return data["close_target"]

        data_batch = {
            "close_target": jnp.linspace(0.5, 1.0, T, dtype=jnp.float32),
        }

        loss = calibration_loss(
            {}, data_batch, perfect_model, jax.random.key(0)
        )
        assert float(loss) < 1e-5

    def test_loss_differentiable(self):
        """calibration_loss is differentiable w.r.t. a parameter."""
        T = 5

        def model_fn(params, data, key):
            scale = params["scale"]
            return data["close_target"] * scale

        data_batch = {
            "close_target": jnp.ones(T, dtype=jnp.float32) * 0.5,
        }

        def loss_wrapper(scale):
            return calibration_loss(
                {"scale": scale}, data_batch, model_fn, jax.random.key(0)
            )

        grad = jax.grad(loss_wrapper)(jnp.float32(1.0))
        assert jnp.isfinite(grad)
