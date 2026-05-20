# stochax-market

Stochastic PDE-based stock market simulation, calibration, and prediction using JAX.

**Version 0.5.0** — Advective SPDE formulation with learnable noise and
drift scales, rollout-based shift-variance calibration, and capped GARCH
persistence. See [`docs/CHANGELOG.md`](docs/CHANGELOG.md) for the full
v0.2 → v0.5 evolution.

## Overview

`stochax-market` implements an advective stochastic PDE (Fokker–Planck-style)
for NIFTY50 equity price evolution, with stochastic volatility supplied by
a calibrated GARCH(1,1). The model evolves a probability-like field
`u(x, t)` over a 1D normalised price domain `[0, 1]`; the predicted price
is the centre of mass of `u` rescaled by the price-scale `L`.

Key model components:

- **Advective SPDE** — `∂u/∂t = κ∇²u − μ_scale·drift·∂u/∂x − σ_scale·σ(x)·∂u/∂x·Ẇ`.
  The drift and noise terms shift `u` along the price axis (advection);
  earlier versions used a multiplicative `μ·u + σ·u·dW` formulation which
  could not move the centre of mass and produced flat forecasts.
- **GARCH(1,1) volatility** with a hard persistence cap (α + β ≤ 0.97) to
  prevent unit-root pathologies during calibration.
- **Pseudo-spectral diffusion** via `exponax`, with κ kept small (1e-4)
  so the Gaussian initial condition doesn't broaden enough to be affected
  by the periodic boundaries over a 150-step forecast horizon.
- **Karhunen–Loève spatial noise** with eigenvalues `λᵢ = (σ_emp / i)²`,
  anchored to empirical daily log-return volatility.
- **Differentiable calibration** through Optimistix BFGS, with a four-term
  loss: teacher-forced MSE + directional + GARCH variance-ratio anchor +
  rollout-based shift-variance penalty.

The full mathematical specification is in [`docs/model.md`](docs/model.md).

## Installation

```bash
uv sync
uv pip install -e .
```

This registers seven CLI commands: `stochax-simulate`, `stochax-fit`,
`stochax-predict`, `stochax-merge`, `stochax-diagnose`,
`stochax-visualize`, `stochax-export`.

## Data setup

`stochax-market` reads per-symbol historical data from local CSV files
under `./data/` (relative to the directory you run commands from). The
expected source is the V1.1.0 fetcher output from the companion
[`upstox-historical`](https://github.com/KosmicByte/upstox-historical)
project:

```
~/github/TradeEngine/
├── data/
│   ├── RELIANCE.csv
│   ├── TCS.csv
│   └── ...
```

Required schema (V1.1.0 capitalised columns):

```
Date, Symbol, Series, Prev Close, Open, High, Low, Close, Volume,
VWAP, Turnover, Trades, Deliverable Volume, %Deliverble
```

Dates may be tz-aware or tz-naive — the loader normalises both to
tz-naive calendar dates. Missing files raise `FileNotFoundError` with
the absolute path that was expected.

## What this model is and is not for

**Strong** (validated on RELIANCE 2014–2026):

- **Volatility forecasting** — fitted GARCH closely tracks 30-day rolling
  realised vol (ρ ≈ 0.79), and unconditional GARCH variance matches
  realised within 5% after calibration.
- **Confidence-band quantification** — 90% CI bands from
  `stochax-predict` reflect proper one-step-ahead and multi-step
  rollout variance (predicted-vs-actual std ratio ≈ 0.6 on 150-day
  backtests with `K=150` rollout-trained sigma_scale).
- **Directional accuracy at the daily level** — ~63% consecutive
  directional accuracy on the post-fit backtest window.

**Weak**:

- **Point prediction of long-horizon trends in regime-shifted markets** —
  the SPDE drift term anchors to a scalar mean (full history or windowed).
  When the recent regime diverges sharply from the historical mean (e.g.
  RELIANCE Apr 2026: +15% historical drift vs −18% recent-60d), the
  forecast mean lags. Theil's U against a naive random walk currently
  lands in the 2.0–2.5 range over 150-day horizons. Use Workflow C
  backtests on your own data to confirm regime suitability before relying
  on point forecasts.

For point forecasts in shifting regimes, default `stochax-simulate` and
`stochax-predict` to a windowed drift (`--drift-window 60` or `126`) — the
diagnostics report explicitly flags regime divergence when present.

## Quick start

The end-to-end pipeline is **fit → simulate → merge → diagnose → visualize**,
with `predict` as the ensemble alternative for confidence-banded forecasts.

For the full workflow including daily-refresh patterns and backtests, see
[`docs/workflow.md`](docs/workflow.md).

### 1. Calibrate model parameters

```bash
stochax-fit --symbol RELIANCE --n-steps 1000 --output params.pkl
```

Runs BFGS for up to 1000 steps on the full history (3000+ days for
RELIANCE). Prints a pre-fit diagnostic block (parameter init, GARCH
unconditional vs realised variance, drift, loss decomposition with
weighted shares) and a post-fit block (final state, per-component loss
breakdown, improvement-vs-init, rollout diff-std ratio).

The fit calibrates five parameters:

| Parameter      | Role                                    | Typical fitted value (RELIANCE) |
|----------------|-----------------------------------------|---------------------------------|
| `mu_scale`     | Drift advection strength                | ~0.10                           |
| `sigma_scale`  | Noise advection strength                | ~3–6                            |
| `garch_omega`  | GARCH base variance                     | ~1e-5                           |
| `garch_alpha`  | ARCH coefficient                        | ~0.12                           |
| `garch_beta`   | GARCH coefficient                       | ~0.85 (clipped at 0.97 − α)     |

Useful flags:

- `--training-window N` — fit on the last N days only (default: full history)
- `--quiet / -q` — suppress the diagnostic preamble/postamble blocks

The fit writes a pickled dict with `spde`, `garch`, `L`, and `loss_info`
to the `--output` path.

### 2. Simulate a forward trajectory

```bash
stochax-simulate --symbol RELIANCE --steps 150 --params params.pkl \
                 --output RELIANCE_sim.csv
```

Single deterministic-seed forward roll from the last historical close.
Writes `step, predicted_price` columns. For Monte Carlo ensembles with
CI bands, use `stochax-predict` instead.

Useful flags:

- `--drift-window N` — base the forward drift on the last N days' mean,
  rather than full-history. Try `60` or `126` when the diagnostics flag
  a regime shift.

### 3. Merge predictions with realised actuals

```bash
stochax-merge --sim-csv RELIANCE_sim.csv --symbol RELIANCE
```

Augments the sim CSV in place with `actual_price` and `date` columns, so
the downstream diagnostics and visualisation tools can score against
reality. Idempotent — rerun any time to pick up newly-realised days.

For backtests, anchor the forecast at a specific historical date:

```bash
stochax-merge --sim-csv RELIANCE_sim.csv --symbol RELIANCE \
              --start-date 21/03/26
```

### 4. Diagnose forecast quality

```bash
stochax-diagnose --symbol RELIANCE \
                 --sim-csv RELIANCE_sim.csv \
                 --params  params.pkl \
                 --out     RELIANCE_diagnostics.md
```

Seven diagnostic sections (Forecast Accuracy, Directional, Variance,
Residual, GARCH Health, Drift Calibration, Calibration Convergence). Each
finding has a status (`ok`/`warn`/`fail`) and a one-line diagnosis.

See [`docs/diagnostics.md`](docs/diagnostics.md) for the per-metric
threshold reference.

### 5. Predict with Monte Carlo confidence bands

```bash
stochax-predict --symbol RELIANCE --horizon 21 --params params.pkl --samples 100
```

Runs 100-path Monte Carlo via `jax.vmap`. Returns mean prediction, 5%
and 95% CI bands, and all sample paths. Use this rather than
`stochax-simulate` when you want uncertainty quantification.

Useful flags:

- `--drift-window N` — same regime-window control as `stochax-simulate`.
- `--samples K` — number of MC paths (default 100).

### 6. Visualise results

```bash
stochax-visualize --symbol RELIANCE \
                  --sim-csv RELIANCE_sim.csv \
                  --params  params.pkl \
                  --horizon 21 --recent-n 60 \
                  --out-dir plots/
```

Generates up to seven plots depending on inputs provided:

| Plot                  | Inputs needed       | Purpose                                  |
|-----------------------|---------------------|------------------------------------------|
| `historical`          | symbol              | Close + VWAP time series                 |
| `log_returns`         | symbol              | Return distribution + kurtosis           |
| `volatility`          | symbol              | 30-day rolling annualised vol            |
| `sim_vs_actual`       | + sim CSV           | Simulation overlaid on most recent N     |
| `actual_vs_predicted` | + merged sim CSV    | Forecast accuracy + MAPE on realised     |
| `prediction`          | + params            | Recent N days + horizon-day MC bands     |
| `garch_fit`           | + params            | GARCH conditional vol vs 30d realised    |

## Project structure

```
src/stochax_market/
├── data/
│   ├── loader.py       # CSV loader; tz-normalisation
│   └── features.py     # OHLCV → JAX arrays; log-returns; u0 stack
├── model/
│   ├── spde.py         # Advective SPDEStepper (Equinox module)
│   ├── volatility.py   # GARCH(1,1) with persistence cap
│   ├── noise.py        # Karhunen-Loève Q-Wiener noise
│   └── initial.py      # Gaussian price-to-field mapping
├── calibration/
│   ├── fit.py          # Optimistix BFGS + 4-term penalised loss
│   └── loss.py         # field_mean (centre-of-mass) + reference loss
├── simulate.py         # Deterministic-seed forward rollout
├── predict.py          # Monte Carlo forecast with CI bands
├── merge.py            # Sim CSV ↔ realised actuals
├── diagnostics.py      # 7-section quality report
├── visualize.py        # Plotly diagnostic plots
└── cli.py              # Typer CLI for all 7 commands

data/                   # (not in repo) Place V1.1.0 CSVs here
└── RELIANCE.csv
```

## Documentation

- [Workflow Guide](docs/workflow.md) — first-time run, daily refresh, backtests
- [Model Specification](docs/model.md) — mathematical formulation and v0.5 parameter table
- [Calibration Method](docs/calibration.md) — loss components and why each weight has its current value
- [Diagnostics Reference](docs/diagnostics.md) — per-finding metric reference
- [Architecture Design](docs/design.md) — components and rationale
- [Problem Statement](docs/problem.md) — motivation and scope
- [API Reference](docs/api.md) — function signatures
- [Changelog](docs/CHANGELOG.md) — version history

## Testing

```bash
pytest tests/
```
