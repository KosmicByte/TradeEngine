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

## Data Setup

`stochax-market` reads per-symbol historical data from local CSV files
under `./data/` (relative to the directory you run commands from). The
expected source is the V1.1.0 fetcher output from the companion
[`upstox-historical`](https://github.com/KosmicByte/upstox-historical)
project. Place fetched files like so:

```
~/github/TradeEngine/
├── data/
│   ├── RELIANCE.csv
│   ├── TCS.csv
│   └── ...
```

Each CSV must use the V1.1.0 capitalised schema:

```
Date, Symbol, Series, Prev Close, Open, High, Low, Close, Volume,
VWAP, Turnover, Trades, Deliverable Volume, %Deliverble
```

Dates may be tz-aware (e.g. `2026-05-08 00:00:00+05:30`) or tz-naive ISO —
the loader normalises both to tz-naive local calendar dates so downstream
matching against business-day forecast grids works correctly. (Earlier
versions silently failed on tz-aware inputs; fixed in v0.4.)

If a required file is missing, `load_stock(symbol)` raises
`FileNotFoundError` with the absolute path it expected.

## Quick Start

The end-to-end pipeline is **fit → simulate → merge → diagnose → visualize**,
with `predict` as an alternative ensemble path for confidence-banded
forecasts.

> 📖 For the full workflow guide including daily-refresh patterns,
> backtesting, and forecast rotation, see
> [docs/workflow.md](docs/workflow.md).

### 1. Calibrate model parameters

```bash
stochax-fit --symbol RELIANCE --n-steps 5 --output params.pkl
```

Writes a pickled dict containing the fitted `SPDEStepper`,
`GARCHVolatility`, and price scale `L`. The `--n-steps` flag is honoured
(no longer silently capped — fixed in v0.2).

### 2. Simulate a stock trajectory

```bash
stochax-simulate --symbol RELIANCE --steps 10
```

Writes `RELIANCE_sim.csv` with columns `step` and `predicted_price`.

### 3. Merge predictions with actuals

```bash
stochax-merge --sim-csv RELIANCE_sim.csv --symbol RELIANCE
```

Augments the sim CSV in place with two new columns — `actual_price`
(realised Close prices pulled from the V1.1.0 data in `./data/`) and
`date` (business-day grid anchored at the day after the last historical
date) — to match the schema consumed by `plot_actual_vs_predicted` and
`analyze`. Future or holiday steps appear as NaN.

The merge step is **idempotent**: rerun any time and it pulls in
newly-realised actuals without leaving stale NaNs.

To anchor the forecast at a specific date instead of "next business day
after history" (useful for backtests):

```bash
stochax-merge --sim-csv RELIANCE_sim.csv --symbol RELIANCE --start-date 18/04/26
```

### 4. Diagnose forecast quality

```bash
stochax-diagnose --symbol RELIANCE \
  --sim-csv RELIANCE_sim.csv \
  --params  params.pkl \
  --out     RELIANCE_diagnostics.md
```

Runs seven diagnostic sections — forecast accuracy, directional
performance, variance diagnostics, residual analysis, GARCH health,
drift calibration, and calibration convergence — and emits a structured
Markdown report. Each finding carries a status (`ok` / `warn` / `fail`)
and a one-line diagnosis. The Variance Diagnostics section explicitly
catches regressions of the constant-drift and frozen-volatility bugs
fixed in v0.2.

If `--out` is omitted, the full report streams to stdout. See
[docs/diagnostics.md](docs/diagnostics.md) for the metric-by-metric
breakdown.

### 5. Predict future prices (ensemble alternative)

```bash
stochax-predict --symbol RELIANCE --horizon 3 --params params.pkl
```

Runs 100-path Monte Carlo via `jax.vmap` and returns a dict with
`mean_prediction`, `lower_ci`, `upper_ci`, and `all_samples`. v0.2 fixes
two bugs in the forecast path: GARCH volatility is now properly iterated
forward (not held constant at last sigma) and drift uses the historical
mean (not the noisy last log-return).

### 6. Visualize results

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
├── data/             # Dataset loading and feature engineering
├── model/            # SPDE stepper, GARCH volatility, noise, initial conditions
├── calibration/      # Loss functions and Optimistix-based fitting
├── simulate.py       # Forward simulation entrypoint
├── predict.py        # Ensemble forecasting entrypoint (Monte Carlo + CIs)
├── merge.py          # Joins simulate output with V1.1.0 actuals
├── diagnostics.py    # Forecast quality analysis and report generation
├── visualize.py      # Plotly diagnostic plots
└── cli.py            # Typer CLI commands

data/                 # (not in repo) Place V1.1.0 fetcher CSVs here
└── RELIANCE.csv      # One file per symbol
```

## Documentation

- [Workflow Guide](docs/workflow.md) — first-time run, daily refresh, backtesting, rotation
- [Model Derivation](docs/model.md) — mathematical foundations
- [Architecture Design](docs/design.md) — system architecture
- [Problem Statement](docs/problem.md) — motivation and limitations
- [API Reference](docs/api.md) — function signatures
- [Diagnostics Module](docs/diagnostics.md) — per-finding reference

## Testing

```bash
pytest tests/
```
