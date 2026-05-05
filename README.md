# stochax-market

Stochastic PDE-based stock market simulation, calibration, and prediction using JAX.

## Overview

`stochax-market` implements a GARCH-diffusion SPDE model for simulating and predicting
NIFTY50 stock prices. The model extends Geometric Brownian Motion (GBM) with:

- **Stochastic volatility** via GARCH(1,1) diffusion
- **Spatial PDE structure** solved with pseudo-spectral methods (exponax)
- **Differentiable calibration** using Optimistix solvers
- **JAX-native** computation with JIT, vmap, and grad support

## Installation

```bash
uv sync
```

## Quick Start

The end-to-end pipeline is **fit → simulate → merge → visualize**, with `predict`
as an alternative ensemble path for confidence-banded forecasts.

### 1. Calibrate model parameters

```bash
stochax-fit --symbol RELIANCE --n-steps 5 --output params.pkl
```

Writes a pickled dict containing the fitted `SPDEStepper`, `GARCHVolatility`,
and price scale `L`. The `--n-steps` flag is honoured (no longer silently
capped — fixed in v0.2).

### 2. Simulate a stock trajectory

```bash
stochax-simulate --symbol RELIANCE --steps 10
```

Writes `RELIANCE_sim.csv` with columns `step` and `predicted_price`.

### 3. Merge predictions with actuals

```bash
stochax-merge --sim-csv RELIANCE_sim.csv --symbol RELIANCE
```

Augments the sim CSV in place with two new columns — `actual_price` (realised
Close prices pulled from the V1.1.0 `upstox-historical` fetch) and `date`
(business-day grid anchored at the day after the last training date) — to
match the schema consumed by `plot_actual_vs_predicted`. Future / holiday
steps appear as NaN.

The merge step is **idempotent**: rerun it any time and it pulls in
newly-realised actuals without leaving stale NaNs.

To anchor the forecast at a specific date instead of "next business day after
history" (useful for backtests):

```bash
stochax-merge --sim-csv RELIANCE_sim.csv --symbol RELIANCE --start-date 18/04/26
```

### 4. Predict future prices (ensemble alternative)

```bash
stochax-predict --symbol RELIANCE --horizon 3 --params params.pkl
```

Runs 100-path Monte Carlo via `jax.vmap` and returns a dict with
`mean_prediction`, `lower_ci`, `upper_ci`, and `all_samples`. v0.2 fixes two
bugs in the forecast path: GARCH volatility is now properly iterated forward
(not held constant at last sigma) and drift uses the historical mean (not
the noisy last log-return).

### 5. Visualize results

```bash
# Minimum — historical + log-returns + volatility only
stochax-visualize --symbol RELIANCE --out-dir plots/

# With simulation overlay (also auto-generates actual-vs-predicted if the
# CSV has been merged with actuals)
stochax-visualize --symbol RELIANCE \
  --sim-csv RELIANCE_sim.csv \
  --out-dir plots/

# Full suite — all 7 plots including prediction band + GARCH fit
stochax-visualize --symbol RELIANCE \
  --sim-csv RELIANCE_sim.csv \
  --params params.pkl \
  --horizon 21 \
  --recent-n 60 \
  --out-dir plots/
```

The visualize step produces (depending on inputs provided):

| Plot                       | Inputs needed              | Purpose                                  |
|----------------------------|----------------------------|------------------------------------------|
| `historical`               | symbol                     | Full Close + VWAP time series            |
| `log_returns`              | symbol                     | Return distribution + kurtosis           |
| `volatility`               | symbol                     | 30-day rolling annualised vol            |
| `sim_vs_actual`            | + sim CSV                  | Simulation overlaid on most recent N     |
| `actual_vs_predicted`      | + merged sim CSV           | Forecast accuracy with MAPE on realised  |
| `prediction`               | + params                   | Recent N days + horizon-day forecast band|
| `garch_fit`                | + params                   | GARCH conditional vol vs 30d realised    |

## Project Structure

```
src/stochax_market/
├── data/          # Dataset loading and feature engineering
├── model/         # SPDE stepper, GARCH volatility, noise, initial conditions
├── calibration/   # Loss functions and Optimistix-based fitting
├── simulate.py    # Forward simulation entrypoint
├── predict.py     # Ensemble forecasting entrypoint (Monte Carlo + CIs)
├── merge.py       # Joins simulate output with V1.1.0 actuals
├── visualize.py   # Plotly diagnostic plots
└── cli.py         # Typer CLI commands
```

## Documentation

- [Model Derivation](docs/model.md)
- [Architecture Design](docs/design.md)
- [Problem Statement](docs/problem.md)
- [API Reference](docs/api.md)

## Testing

```bash
pytest tests/
```
