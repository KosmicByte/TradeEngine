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

### Simulate a stock trajectory

```bash
stochax-simulate --symbol RELIANCE --steps 10
```

### Calibrate model parameters

```bash
stochax-fit --symbol RELIANCE --n-steps 5 --output params.pkl
```

### Predict future prices

```bash
stochax-predict --symbol RELIANCE --horizon 3 --params params.pkl
```

#### Visualize results

```bash

# Minimum — historical + log-returns + volatility only
stochax-visualize --symbol RELIANCE --out-dir plots/

# With simulation overlay
stochax-visualize --symbol RELIANCE \
  --sim-csv RELIANCE_sim.csv \
  --out-dir plots/

# Full suite — all 6 plots including prediction + GARCH fit
stochax-visualize --symbol RELIANCE \
  --sim-csv RELIANCE_sim.csv \
  --params params.pkl \
  --horizon 21 \
  --recent-n 60 \
  --out-dir plots/

```
## Project Structure

```
src/stochax_market/
├── data/          # Dataset loading and feature engineering
├── model/         # SPDE stepper, GARCH volatility, noise, initial conditions
├── calibration/   # Loss functions and Optimistix-based fitting
├── simulate.py    # Forward simulation entrypoint
├── predict.py     # Forecasting entrypoint
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
