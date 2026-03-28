"""Tests for data loading and feature engineering."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from stochax_market.data.features import encode_features


class TestEncodeFeatures:
    """Tests for encode_features."""

    def _make_sample_df(self, n: int = 20) -> pd.DataFrame:
        """Create a sample DataFrame mimicking NIFTY50 data."""
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        rng = np.random.default_rng(42)
        prices = 1000.0 + np.cumsum(rng.normal(0, 10, n))

        return pd.DataFrame({
            "Date": dates,
            "Prev Close": np.roll(prices, 1),
            "Open": prices + rng.normal(0, 5, n),
            "High": prices + abs(rng.normal(0, 10, n)),
            "Low": prices - abs(rng.normal(0, 10, n)),
            "Close": prices,
            "VWAP": prices + rng.normal(0, 2, n),
        })

    def test_encode_features_keys(self):
        """encode_features returns all expected keys."""
        df = self._make_sample_df()
        features = encode_features(df, nx=64)

        expected_keys = {
            "u0", "drift", "vwap_target", "close_target",
            "price_range", "log_returns", "dates", "seasonal", "price_scale"
        }
        assert expected_keys.issubset(set(features.keys()))

    def test_encode_features_shapes(self):
        """encode_features returns arrays with correct shapes."""
        n = 20
        nx = 64
        df = self._make_sample_df(n)
        features = encode_features(df, nx=nx)

        # First row has NaN from roll, so T may be n-1 or n
        T = features["u0"].shape[0]
        assert features["u0"].shape == (T, nx)
        assert features["drift"].shape == (T,)
        assert features["vwap_target"].shape == (T,)
        assert features["close_target"].shape == (T,)
        assert features["price_range"].shape == (T, 2)
        assert features["log_returns"].shape == (T,)
        assert features["dates"].shape == (T,)
        assert features["seasonal"].shape == (T, 2)

    def test_encode_features_dtypes(self):
        """All JAX arrays have float32 dtype."""
        df = self._make_sample_df()
        features = encode_features(df, nx=64)

        for key in ["u0", "drift", "vwap_target", "close_target",
                     "log_returns", "seasonal"]:
            assert features[key].dtype == jnp.float32, f"{key} has wrong dtype"

    def test_encode_features_missing_column_raises(self):
        """encode_features raises ValueError for missing columns."""
        df = pd.DataFrame({"Date": [1], "Prev Close": [100]})
        with pytest.raises(ValueError, match="Missing required column"):
            encode_features(df)

    def test_u0_is_nonnegative(self):
        """Initial conditions u0 should be non-negative."""
        df = self._make_sample_df()
        features = encode_features(df, nx=64)
        assert jnp.all(features["u0"] >= 0.0)

    def test_seasonal_bounded(self):
        """Seasonal features should be in [-1, 1]."""
        df = self._make_sample_df()
        features = encode_features(df, nx=64)
        assert jnp.all(features["seasonal"] >= -1.0)
        assert jnp.all(features["seasonal"] <= 1.0)
