# Architecture Design

## Data Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  kagglehub   │────▶│  loader.py   │────▶│ features.py  │
│  (download)  │     │  (CSV parse) │     │  (encode)    │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                     ┌────────────────────────────┘
                     ▼
    ┌────────────────────────────────────────────────────┐
    │                 Model Pipeline                      │
    │                                                    │
    │  ┌──────────┐  ┌──────────────┐  ┌──────────────┐ │
    │  │ initial  │  │  volatility  │  │    noise     │ │
    │  │  .py     │  │    .py       │  │     .py      │ │
    │  │ u₀(x)   │  │  GARCH σ(t)  │  │  KL W(x,t)  │ │
    │  └────┬─────┘  └──────┬───────┘  └──────┬───────┘ │
    │       │               │                  │         │
    │       └───────┬───────┴──────────────────┘         │
    │               ▼                                    │
    │       ┌──────────────┐                             │
    │       │   spde.py    │                             │
    │       │  SPDEStepper │  exponax diffusion          │
    │       │  + noise     │  + multiplicative terms     │
    │       └──────┬───────┘                             │
    └──────────────┼─────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
 ┌──────────────┐     ┌──────────────┐
 │ simulate.py  │     │ calibration/ │
 │ (forward)    │     │  fit.py      │
 └──────┬───────┘     │  loss.py     │
        │             └──────┬───────┘
        ▼                    ▼
 ┌──────────────┐     ┌──────────────┐
 │  predict.py  │     │  Optimistix  │
 │  (forecast)  │     │  BFGS solver │
 └──────────────┘     └──────────────┘
```

## Rationale for exponax

[exponax](https://github.com/Ceyron/exponax) provides pseudo-spectral exponential
time differencing (ETD) solvers for PDEs in JAX:

- **Spectral accuracy** for smooth solutions on periodic domains
- **ETD handles stiff diffusion** — the linear κ∇²u term is solved exactly in
  Fourier space, allowing larger timesteps without stability issues
- **JAX-native** — fully differentiable, JIT-compilable, and vmap-compatible
- **Clean API** — `Diffusion(num_spatial_dims=1, ...)` creates a stepper that
  operates on (C, N) arrays with a single function call

We use `exponax.stepper.Diffusion` for the linear part and apply multiplicative
noise/drift in physical space after each diffusion step.

## Rationale for Optimistix

[Optimistix](https://docs.kidger.site/optimistix) provides differentiable
optimization solvers native to JAX:

- **Multiple solver backends**: BFGS, Levenberg-Marquardt, Gauss-Newton, etc.
- **Pure JAX** — all solvers are JIT-compatible and differentiable
- **Equinox integration** — works naturally with Equinox modules as pytrees
- **Adjoint methods** — supports implicit differentiation through the solve

We use `optimistix.BFGS` for minimizing the MSE loss between predicted and
observed Close/VWAP prices.

## Equinox PyTree Parameter Design

All trainable parameters are stored in Equinox modules:

- **SPDEStepper**: `raw_mu_scale` (constrained via softplus; `kappa` is a fixed hyperparameter passed as a concrete `float` to exponax at construction time and frozen inside `diffusion_stepper`)
- **GARCHVolatility**: `raw_omega`, `raw_alpha`, `raw_beta` (softplus + rescaling)

This design ensures:
1. Parameters are valid JAX pytrees (no mutable state)
2. Positivity constraints are automatically enforced
3. Gradient computation works out-of-the-box via `jax.grad`
4. Serialization via pickle is straightforward

## Extensibility Guide

### Adding New Features

1. Add feature extraction to `data/features.py` `encode_features()`
2. Add the feature as an input to `model/spde.py` `step()` method
3. Update `calibration/loss.py` if the feature affects the loss

### Adding New Model Terms

1. Define the new term (e.g., a nonlinear reaction) in a new file under `model/`
2. Integrate it into `SPDEStepper.step()` as an additional physical-space correction
3. If the term has trainable parameters, make it an Equinox module

### Adding New Solvers

1. Import the solver from Optimistix (e.g., `optimistix.LevenbergMarquardt`)
2. Replace the solver in `calibration/fit.py`
3. Adjust convergence tolerances as needed
