# Problem Statement

## Objective

Predict daily Close and VWAP prices for NIFTY50 stocks using a stochastic PDE
(SPDE) model that captures the essential statistical properties of financial
time series.

## Why Linear / ARMA Models Are Insufficient

Classical linear time series models (ARMA, ARIMA) assume:
- Constant conditional variance (homoscedasticity)
- Gaussian innovations
- Linear dependence structure

However, financial returns exhibit:

1. **Non-constant variance** — The ACF of squared returns |X_t|² and |X_t| decays
   slowly, indicating long-range dependence in volatility. This violates the
   homoscedasticity assumption of ARMA.

2. **Fat-tailed innovations** — PACF/ACF analysis shows that residuals from
   fitted ARMA models have excess kurtosis, suggesting the Gaussian assumption
   is violated.

3. **Nonlinear dynamics** — Volatility responds asymmetrically to positive vs.
   negative shocks (leverage effect), which is inherently nonlinear.

## Why GARCH Alone Is Insufficient

GARCH(1,1) addresses volatility clustering but has limitations:

1. **No spatial structure** — GARCH operates on a single scalar time series.
   It cannot model the joint evolution of a distribution over price space.

2. **No state-space field** — There is no concept of a "price field" that
   evolves continuously. GARCH models σ²_t but not the full probability
   distribution of future prices.

3. **Limited forecasting** — Multi-step forecasts from GARCH converge quickly
   to the unconditional variance, providing little information beyond a few
   steps ahead.

4. **No spatial correlations** — Cross-sectional structure (e.g., intraday
   high/low/VWAP relationships) cannot be captured by a univariate model.

## The SPDE Approach

Our SPDE model addresses these limitations by:

1. Evolving a spatial probability field u(x, t) over a 1D price grid
2. Using GARCH-diffusion for stochastic volatility in the PDE
3. Leveraging pseudo-spectral methods for efficient and accurate simulation
4. Enabling differentiable calibration via JAX + Optimistix

## Known Limitations

1. **1D spatial domain** — The current model uses a 1D price grid. Multi-asset
   correlations would require higher-dimensional grids.

2. **Periodic boundary conditions** — exponax uses periodic BCs, which may
   introduce artifacts at domain boundaries for non-periodic price fields.

3. **GARCH simplicity** — GARCH(1,1) may not capture all volatility dynamics.
   Extensions like EGARCH or GJR-GARCH could improve the model.

4. **Calibration difficulty** — The SPDE has many interacting parameters, and
   the loss landscape may have local minima.

5. **No jump processes** — The model does not include jump-diffusion terms
   that could capture sudden price discontinuities (e.g., earnings surprises).

6. **Data limitations** — Only daily OHLCV data is used; intraday information
   could improve predictions.
