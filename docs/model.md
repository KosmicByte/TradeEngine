# Mathematical Model — v0.5 Advective SPDE

This is the formal specification of the `stochax-market` v0.5 model. It
replaces the role of `test_model.pdf` from the early multiplicative-form
versions; that PDF's fitted parameter values are obsolete and should not
be relied on.

The model evolves a probability-like field `u(x, t)` over a 1D normalised
domain `[0, 1]`, where the predicted price at time `t` is the centre of
mass of `u(·, t)` rescaled by the price-scale `L`. Calibration recovers
five differentiable parameters from historical OHLCV data using
Optimistix BFGS on a penalised one-step + rollout loss.

---

## 1. From GBM to advective SPDE

### Geometric Brownian Motion baseline

The classical reference is Geometric Brownian Motion:

```
dP_t = μ P_t dt + σ P_t dB_t,    X_t = log(P_t / P_{t-1})
```

GBM's central failure for daily equity returns is that `σ` is treated as
constant. Real returns exhibit:

- **Volatility clustering** — `|X_t|` and `X_t²` have slowly-decaying
  autocorrelation; large moves cluster.
- **Excess kurtosis** — RELIANCE 12-year daily returns have excess
  kurtosis ≈ 8; tails are much heavier than Gaussian.
- **Leverage effect** — negative shocks raise future vol more than
  positive shocks of the same size.

GARCH(1,1) is the standard fix for the variance clustering. Coupling
GARCH to a spatial PDE then adds a price *distribution* (rather than
just a price *value*), which is what enables consistent CI-band
forecasting.

### The model

In its current advective form (v0.4+):

```
∂u/∂t = κ ∇²u − μ_scale · drift(t) · ∂u/∂x − σ_scale · σ(x, t) · ∂u/∂x · Ẇ(x, t)
```

with:

| Symbol         | Meaning                                                        |
|----------------|----------------------------------------------------------------|
| `u(x, t)`      | Probability-like field on `x ∈ [0, 1]`                         |
| `κ`            | Diffusion coefficient (fixed, small; see §3)                   |
| `μ_scale`      | Drift advection scale (learnable, softplus-constrained > 0)    |
| `drift(t)`     | Per-step drift signal (historical log-return, calibration) or scalar mean (forecast) |
| `σ_scale`      | Noise advection scale (learnable, softplus-constrained > 0)    |
| `σ(x, t)`      | Spatial volatility field, `σ(x, t) = σ_t · n_x · (1 + 0.1·sin(πx))` |
| `σ_t`          | GARCH(1,1) conditional volatility at time `t`                  |
| `Ẇ(x, t)`      | Cylindrical Q-Wiener increment, KL-discretised                 |

The predicted price is

```
p̂(t) = L · ∫₀¹ x · u(x, t) dx / ∫₀¹ u(x, t) dx
```

— the mass-normalised centre of mass times the price-scale `L = max(High)`
across the full dataset. The mass normalisation handles small mass loss
from the non-negativity floor on `u` (see §5).

### Why advective and not multiplicative

Versions v0.2–v0.3 used a multiplicative form:

```
∂u/∂t = κ ∇²u + μ_scale · drift · u + σ(x) · u · Ẇ
```

Multiplicative noise on a narrow Gaussian modulates amplitude pointwise
but does not shift the peak; equivalently, `∫ x · (σ·u·dW) dx` averages
to zero to leading order for a symmetric Gaussian, so the centre of mass
of `u` only diffuses, it doesn't random-walk. RELIANCE backtests in May
2026 produced a predicted-vs-actual std ratio of 0.124 (target ~1) even
at correctly calibrated GARCH — the SPDE was effectively deterministic
even though it had a noise term.

The advective form `−σ·∂u/∂x·dW` is the natural Fokker–Planck-style
counterpart of the underlying price process `dX = μ·dt + σ·dW`: u
translates with the drift and random-walks with the noise. Centre of
mass moves; CI bands fan; calibration becomes well-posed.

---

## 2. GARCH(1,1) volatility

The conditional variance process is the standard GARCH(1,1):

```
σ²_t = ω + α X²_{t-1} + β σ²_{t-1}
```

Parameters are stored in unconstrained space and transformed:

```
ω  = softplus(raw_omega)
α  = 0.05 + 0.15 · sigmoid(raw_alpha)         → α ∈ [0.05, 0.20]
β  = clip(0.50 + 0.40·sigmoid(raw_beta),  max = 0.97 − α)
                                              → β ∈ [0.50, 0.77]
```

The α + β ≤ 0.97 cap is a **hard constraint** (June 2026, v0.4). It
prevents the unit-root corner of the loss landscape that earlier fits
exploited:

When the rollout-based shift-variance penalty creates pressure for
bursty volatility, BFGS otherwise pushes `α + β → 0.999` and shrinks
`ω → 0` to compensate. The result is a GARCH process with vol half-life
of ~700 days — pathological for daily equity, and producing simulation
trajectories that crash off-domain because of compounding tail events.

At the 0.97 cap, the half-life is `ln(0.5) / ln(0.97) ≈ 22.8` days —
empirically reasonable for equity volatility.

### Spatial volatility field

GARCH yields a scalar `σ_t` per timestep. The SPDE step needs a spatial
field. Construct it as:

```
σ_field(x, t) = σ_t · n_x · (1 + 0.1 · sin(π · x))
```

The `n_x = 128` factor counter-scales against the `1/n_x` magnitude of
`u(x, ·)` near the Gaussian peak. The `(1 + 0.1·sin(πx))` modulation gives
non-trivial spatial structure with guaranteed non-zero amplitude
everywhere. The overall amplification compared to the empirical-σ scale
is then absorbed into the learnable `σ_scale` parameter inside the SPDE
step itself, so the fixed factor here is not critical.

---

## 3. Diffusion coefficient

`κ` is **fixed** (not learned) at `1e-4`. The choice matters because:

- Diffusion broadens the initial Gaussian by `σ²(t) = σ²(0) + 2κt`.
- The grid uses periodic boundary conditions (inherited from exponax's
  pseudo-spectral solver).
- Initial prices can be near the domain boundary — RELIANCE's most
  recent close gives `x₀ ≈ 0.86`.

At `κ = 0.01` (the v0.3 default), the Gaussian width grows from ≈0.05
to ≈0.12 over 150 days; significant mass at `x > 1` wraps around to
`x ≈ 0` and pollutes the centre-of-mass calculation. The observed symptom
in v0.3 simulations was the "midpoint regression cliff" — predicted
prices drifted to roughly `L/2` over multi-month horizons.

At `κ = 1e-4`, width growth over 150 days is below 1% (`σ ≈ 0.0501`).
The Gaussian stays where it is unless the advection terms move it; the
periodic-boundary wrap-around is effectively eliminated for any realistic
forecast horizon.

A future enhancement could make `κ` learnable, but the current fixed
value is robust across instruments and horizons up to ~1 year.

---

## 4. Karhunen–Loève noise discretisation

The Q-Wiener process `Ẇ(x, t)` is discretised by truncated KL expansion:

```
W(x) = Σᵢ √λᵢ · ξᵢ · φᵢ(x),    ξᵢ ~ N(0, 1) i.i.d.
φᵢ(x) = √(2/L_grid) · sin(i·π·x / L_grid)        (eigenfunctions of Δ on [0, L_grid])
λᵢ    = (σ_emp / i)²                              (power-law eigenvalues)
```

Defaults:
- `n_modes = 32`
- `σ_emp = 0.015` (empirical daily log-return std for NIFTY large-caps)
- `L_grid = 1.0` (matches the unit-normalised SPDE domain)
- `decay_rate = 2.0` (eigenvalues fall as `i⁻²`)

The KL decay rate is the spatial-regularity parameter: faster decay →
smoother spatial noise. `decay_rate = 2.0` is the trade-off point where
mode 1 carries empirical σ amplitude and higher modes decay fast enough
that the noise field remains in `C^0` (continuous in space).

The per-step noise increment is `√dt · W(x)`, sized to produce price
moves commensurate with empirical daily volatility once the `σ_scale`
factor is fitted.

---

## 5. Spatial gradient and the non-negativity floor

The advection terms need `∂u/∂x`. We use central finite differences with
periodic boundary conditions (matching the exponax stepper's BC):

```
(∂u/∂x)_i = (u_{i+1} − u_{i-1}) / (2 · dx),    dx = L_grid / n_x = 1/128
```

This is second-order accurate in the interior. Periodic BC via
`jnp.roll` is consistent with the diffusion stepper's BC.

After each step, `u` is clipped at zero:

```
u_next = max(u_next, 0)
```

This prevents the field from going negative under large stochastic-
advection excursions. The cost is mass loss; the centre-of-mass formula
(`field_mean` in `loss.py`) divides by the actual mass `∫ u dx + ε`, so
loss is absorbed into the price extraction.

---

## 6. Initial condition

For a price `p` (raw INR), the initial condition is a normalised
Gaussian centred at the normalised coordinate `x₀ = p / L`:

```
u₀(x) = N · exp( −(x − x₀)² / (2 · w²) )
```

with `N` chosen so `∫ u₀ dx = 1`. The default width is `w = 50/L`
(50 INR in raw-price terms, normalised to the unit domain). For
RELIANCE with `L ≈ 1612`, this gives `w ≈ 0.031`.

The width is fixed by `make_initial_condition`; it doesn't enter the
loss explicitly.

---

## 7. Time discretisation

Time step is `Δt = 1 / 252` (one trading day in years). The SPDE
trajectory is rolled forward with `jax.lax.scan`; one diffusion step
(handled by exponax via exponential time differencing in Fourier
space), one drift advection step, one noise advection step per Δt.
Operator splitting is first-order (Lie splitting).

For a forecast horizon of `nt` days, the rollout cost is `O(nt · n_x ·
log n_x)` from the spectral diffusion plus `O(nt · n_x)` from the
advection terms.

---

## 8. Calibration — 5-dim BFGS problem

The trainable parameter vector for `optimistix.minimise` is:

```
y = [raw_mu_scale, raw_sigma_scale, raw_omega, raw_alpha, raw_beta]
```

The full loss is documented in [`calibration.md`](calibration.md). In
summary form:

```
ℓ(y) = 1000 · MSE_TF
     + 0.1  · directional_TF
     + 0.5  · log²( σ²_unc(GARCH) / σ²_realised )
     + 0.3  · log²( var(rollout_diffs) / var(actual_diffs) )
```

where the `_TF` subscript denotes teacher-forced (one-step-ahead)
quantities computed at every t ∈ [0, T), and the rollout shift variance
is computed over a free-running `K = 150` step trajectory from the end
of the training data.

`fit()` returns the fitted `SPDEStepper`, `GARCHVolatility`, and a
`result_info` dict that the diagnostics module consumes.

---

## 9. Forecasting

Forecasting reuses the same SPDE step but starts from `u(x, T)` (the
field at the last historical timestep) and rolls forward with:

- **Drift**: scalar mean over a window — full history by default, or the
  last `drift_window` days when passed. The diagnostics flag a regime
  shift if the recent-60-day annualised drift diverges by more than ~30%
  from the full-history mean; users should respond by setting
  `--drift-window 60` or `126` on `stochax-simulate` /
  `stochax-predict`.
- **Volatility**: GARCH continues forward with σ stochastically updated
  through the fitted (ω, α, β). `predict.py` uses `lax.scan` to roll the
  GARCH recursion stochastically (each Monte Carlo path gets its own σ
  trajectory). `simulate.py` uses the historical σ series as a single
  deterministic path.
- **Noise**: a fresh draw from the KL-Wiener process per path.

For Monte Carlo forecasting with confidence bands, use
`stochax-predict` with `--samples K` (K = 100 typical). For a single
deterministic-seed trajectory (useful for backtesting against actuals),
use `stochax-simulate`.

---

## 10. Fitted parameter table — RELIANCE 2014–2026

For the v0.5 calibration with the full 12-year RELIANCE training window
(3056 daily timesteps), the fitted parameters land near:

| Parameter            | Value             | Notes                                  |
|----------------------|-------------------|----------------------------------------|
| `L` (price scale)    | ₹1611.80          | `max(High)` over training window       |
| `mu_scale`           | ~0.10             | Drift advection scale                  |
| `sigma_scale`        | 3–6               | Noise advection scale (depends on K, weight) |
| `garch_omega`        | ~8e-6             | Base variance                          |
| `garch_alpha`        | ~0.12–0.19        | ARCH (depends on calibration run)      |
| `garch_beta`         | clipped at 0.97−α | GARCH                                  |
| `garch_persistence`  | 0.97              | At the cap                             |
| Unconditional σ      | ~0.017 daily / 27% annualised | Matches realised σ within ±5% |
| Vol half-life        | ~22.8 days        | Empirically realistic for daily equity |

These numbers vary slightly across runs due to RNG seeding and small
trade-offs in the loss landscape but stay in the ranges above. The
backtest expected MAPE is ~3–5% on a 150-day window, with realised
std-ratio ~0.5–0.7. See `docs/CHANGELOG.md` for representative
backtest tables across v0.2–v0.5.

---

## 11. Scope and known limits

The v0.5 model is **a stochastic-process forecaster**, not a regime-
prediction model. Specifically:

- **Variance forecasting and CI quantification**: validated. Unconditional
  GARCH variance matches realised within 5%; rollout std-ratio
  reaches 0.5–0.7 on 150-day backtests.
- **Daily directional accuracy**: ~60–63% in post-fit windows on RELIANCE
  — better than chance, far below trade-relevant edge.
- **Multi-month point prediction in regime-shifted markets**: lags. The
  scalar mean-drift can't span an up-regime and a down-regime
  simultaneously; Theil's U lands in 2.0–2.5 on 150-day backtests when
  the recent regime diverges from the historical mean. Use
  `--drift-window` mitigations on `stochax-simulate` / `stochax-predict`
  to lean into the recent regime.

Future work to address Theil's U > 1: AR(1)-style drift signal (memory
in the drift channel), regime-switching drift, or non-Markov
extensions to the SPDE's deterministic component.

The 1D-spatial domain, the periodic boundary conditions, the GARCH(1,1)
limit on volatility dynamics, and the absence of jump processes are
known structural constraints. See [`docs/problem.md`](problem.md) for
the longer treatment.
