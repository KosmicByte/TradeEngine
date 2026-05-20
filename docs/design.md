# Architecture Design

## Data flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  ./data/     │────▶│  loader.py   │────▶│ features.py  │
│  (V1.1.0     │     │  (CSV parse, │     │  (encode)    │
│   CSVs)      │     │   tz norm)   │     │              │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                     ┌────────────────────────────┘
                     ▼
    ┌────────────────────────────────────────────────────┐
    │                 Model Pipeline                     │
    │                                                    │
    │  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  │
    │  │ initial  │  │  volatility  │  │    noise     │  │
    │  │  .py     │  │    .py       │  │     .py      │  │
    │  │ u₀(x)    │  │ GARCH σ(t)   │  │  KL W(x,t)   │  │
    │  │ Gaussian │  │ with α+β     │  │  λᵢ=(σ_e/i)² │  │
    │  │ at p/L   │  │ cap ≤ 0.97   │  │              │  │
    │  └────┬─────┘  └──────┬───────┘  └──────┬───────┘  │
    │       │               │                  │         │
    │       └───────┬───────┴──────────────────┘         │
    │               ▼                                    │
    │       ┌──────────────────────┐                     │
    │       │      spde.py         │                     │
    │       │ Advective SPDEStepper│                     │
    │       │ exponax diffusion    │                     │
    │       │ + ∂u/∂x advection    │                     │
    │       │ + learnable          │                     │
    │       │   mu_scale,          │                     │
    │       │   sigma_scale        │                     │
    │       └──────┬───────────────┘                     │
    └──────────────┼─────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
 ┌──────────────┐     ┌──────────────┐
 │ simulate.py  │     │ calibration/ │
 │ predict.py   │     │  fit.py      │
 │ (rollouts)   │     │  loss.py     │
 └──────┬───────┘     └──────┬───────┘
        │                    │
        ▼                    ▼
 ┌──────────────┐     ┌──────────────┐
 │  merge.py    │     │  Optimistix  │
 │  diagnostics │     │  BFGS solver │
 │  visualize   │     │  4-term loss │
 └──────────────┘     └──────────────┘
```

## Rationale for exponax

[exponax](https://github.com/Ceyron/exponax) provides pseudo-spectral
exponential time differencing (ETD) solvers for PDEs in JAX:

- **Spectral accuracy** for smooth solutions on periodic domains
- **ETD handles stiff diffusion** — the linear `κ∇²u` term is solved
  exactly in Fourier space, allowing larger timesteps without stability
  issues
- **JAX-native** — fully differentiable, JIT-compilable, vmap-compatible
- **Clean API** — `Diffusion(num_spatial_dims=1, ...)` creates a stepper
  that operates on `(C, N)` arrays via a single function call

In v0.5+ we use a small `κ = 1e-4`, so the spectral solver is barely
doing work — its main role is correctness and JAX-pytree compatibility,
not performance-critical diffusion. The advection terms (which
contribute the actual price dynamics) are applied in physical space
after each spectral step.

## Rationale for Optimistix

[Optimistix](https://docs.kidger.site/optimistix) provides differentiable
optimisation solvers native to JAX:

- **Multiple solver backends** — BFGS, Levenberg-Marquardt, Gauss-Newton
- **Pure JAX** — all solvers are JIT-compatible and differentiable
- **Equinox integration** — solver state and parameters are valid
  pytrees, which keeps `eqx.tree_at(...)` parameter updates clean
- **Adjoint methods** — supports implicit differentiation through the
  solve, useful if calibration is ever nested inside an outer loop

We use `optimistix.BFGS` with `rtol=1e-5, atol=1e-5` to minimise the
4-term penalised loss over a 5-dimensional parameter vector. See
[`calibration.md`](calibration.md) for the loss decomposition.

## Equinox PyTree parameter design

All trainable parameters are stored as fields on Equinox modules, so
they are JAX pytrees and `jax.grad` / `eqx.tree_at` work natively.

**`SPDEStepper`** (`model/spde.py`):

| Field             | Constraint        | Role                                       |
|-------------------|-------------------|--------------------------------------------|
| `raw_mu_scale`    | softplus → > 0    | Drift advection scale                      |
| `raw_sigma_scale` | softplus → > 0    | Noise advection scale (added in v0.5)      |
| `diffusion_stepper` | static (frozen) | exponax `Diffusion(diffusivity=κ)`         |
| `nx`, `dt`, `domain_extent` | static  | Hyperparameters                            |

`κ` is *not* a parameter — it's a hyperparameter baked into
`diffusion_stepper` at construction time. To change it, construct a new
`SPDEStepper`; the old `params.pkl` is incompatible with a different `κ`.

**`GARCHVolatility`** (`model/volatility.py`):

| Field        | Constraint                                | Role           |
|--------------|-------------------------------------------|----------------|
| `raw_omega`  | softplus → > 0                            | Base variance  |
| `raw_alpha`  | `0.05 + 0.15·sigmoid` → [0.05, 0.20]      | ARCH           |
| `raw_beta`   | `clip(0.50 + 0.40·sigmoid, max=0.97−α)` → effectively [0.50, 0.92] | GARCH |

The hard cap on `α + β ≤ 0.97` (see `model.md` §2) is enforced inside
the `beta` property's `jnp.minimum`, so any downstream code reading
`garch.beta` automatically gets the capped value.

## Extensibility

### Adding a new model term

1. Implement the new term inside `SPDEStepper.step()` as an additional
   physical-space correction after the diffusion step.
2. If the term has trainable parameters, add them as Equinox fields
   (with appropriate softplus / sigmoid constraints).
3. Add the new `raw_*` parameter to the BFGS vector `y0` in
   `calibration/fit.py::fit`, mirroring how `raw_sigma_scale` was added
   in v0.5.
4. Update the diagnostic preamble/postamble in `fit.py` to print the
   new parameter so calibration runs are debuggable.

### Adding a new loss term

1. Add the per-component computation to `_penalised_loss_parts` in
   `fit.py`.
2. Add it to `_penalised_loss` and `_weighted_total_from_parts` with a
   `_NEW_WEIGHT` constant at the top of the file.
3. Document the weight choice in `docs/calibration.md` §"Why each term
   has its current weight" — the format is to explain what failure
   mode the term prevents and what empirical observations drove the
   weight value.
4. Include the new term in the preamble/postamble decomposition tables.

### Adding a new diagnostic finding

See `docs/diagnostics.md` §"Architecture and extension". The pattern is
to add a `Finding` to the relevant `_*_findings` function and ensure
the threshold doc in `diagnostics.md` is updated.
