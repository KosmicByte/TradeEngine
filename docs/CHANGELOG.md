# Changelog

All notable changes to `stochax-market`. Format inspired by [Keep a
Changelog](https://keepachangelog.com/), but with diagnostic findings
as the unit of change — every behavioural change here was driven by a
specific diagnostic observation, and the diagnostic motivation is part
of the entry.

## [0.5.0-stable] — June 2026

**Headline.** Advective SPDE formulation with learnable noise and
drift scales, four-term penalised calibration loss, persistence-capped
GARCH, and rollout-based shift-variance penalty matched to the forecast
horizon.

### Added

- `SPDEStepper.raw_sigma_scale` — learnable softplus-constrained noise
  amplification parameter. Calibration vector is now 5-D
  `[raw_mu_scale, raw_sigma_scale, raw_omega, raw_alpha, raw_beta]`.
- `_PERSISTENCE_CAP = 0.97` in `model/volatility.py` — hard upper bound
  on α + β. Prevents the unit-root corner that the rollout penalty
  could otherwise exploit (see v0.4 fix).
- Rollout-based shift-variance penalty in `fit.py`
  (`_rollout_shift_variance_penalty`). Runs a K=150-step free-running
  rollout from `u₀[T−K]` and compares the variance of consecutive
  predicted-price diffs to actuals. K=150 matches the typical forecast
  horizon, so the penalty sees the same range of stochastic excursions
  that the forecast will produce.
- Diagnostic preamble + postamble in `fit.py` with per-component
  weighted loss shares, initial/final rollout diff-std ratios, and an
  automatic warning if BFGS made <0.1% progress.
- `--training-window` and `--quiet` flags on `stochax-fit`.
- `--drift-window` flag on `stochax-simulate` and `stochax-predict`
  for regime-shifted markets.
- `result_info` dict from `fit()` now exposes `mu_scale`, `sigma_scale`,
  `garch_*`, `unc_var_ratio`, `rollout_diff_std_*`, full
  `loss_weights`, and `mode` strings for traceability.

### Changed

- **`SPDEStepper.step()` reformulated** as advective:
  `−μ_scale·drift·∂u/∂x − σ_scale·σ(x)·∂u/∂x·dW`
  in place of the prior multiplicative
  `μ_scale·drift·u + σ(x)·u·dW`. The multiplicative form could not
  shift `u`'s centre of mass and produced near-flat forecasts even at
  correctly calibrated GARCH (RELIANCE backtest May 2026: std ratio
  0.124, Theil's U 1.91). The advective form is the Fokker–Planck
  counterpart of the underlying price process; centre of mass moves
  with the drift and random-walks with the noise.
- **Calibration loss reorganised** to four weighted components:
  `1000·MSE_TF + 0.1·directional_TF + 0.5·log²(unc_var/realised_var) +
  0.3·log²(rollout_diff_var/actual_diff_var)`.
- **Default `kappa` reduced** from `0.01` to `1e-4`. Earlier value
  caused the Gaussian initial condition to broaden enough over a
  150-day rollout that periodic-BC wrap-around polluted
  `field_mean`, dragging forecasts toward the domain midpoint.
- **Teacher-forced calibration** replaces single-trajectory rollout
  in `fit.py::_run_model`. For every t ∈ [0, T) the SPDE is stepped
  one day from `u₀[t]` and compared to `close_target[t]`. The prior
  approach (single 3000-step rollout from `u₀[0]`) produced
  ungradient-able loss landscapes for long T because no SPDE can track
  an equity price 12 years out from a day-0 anchor.
- **Default `n_steps` on `stochax-fit`** bumped from 100 to 1000 to
  match the BFGS step cap. The previous 100 silently capped before
  meaningful convergence.

### Fixed

- `simulate.py` forward drift is now a windowable scalar mean (full
  history by default). Earlier versions used `features["drift"][-nt:]`,
  which fed the literal historical log-return *sequence* in as forward
  drift — i.e. it replayed the last nt days of price history as if they
  were the forecast.
- Default `raw_omega = -11.0` (was −3.0). The earlier default gave
  unconditional σ ≈ 99% annualised, vs realised σ ≈ 27% — a 13× variance
  ratio that the calibration alone couldn't always fix.
- Removed debug print statements from `simulate.py`.

### Validation (RELIANCE 2014–2026, 150-day backtest from 21/03/26)

| Metric                          | v0.2-baseline | v0.5-stable |
|---------------------------------|---------------|-------------|
| GARCH unconditional/realised    | 2492×         | ~1.0        |
| Predicted/actual std ratio      | 0.148         | ~0.5–0.7    |
| Theil's U vs random walk        | 3.117         | ~2.2–2.5    |
| Directional accuracy            | 41.67%        | 60–63%      |
| Bias (mean error)               | +₹51          | ±₹20        |
| Cliff dive in 150-day sim       | yes           | no          |
| Forecast horizon end (₹)        | ~₹900 (cliff) | ~₹1300      |

Theil's U > 1 remains — that's the structural limit of the scalar
mean-drift channel in regime-shifted markets and is the explicit
focus of post-v0.5 work (AR(1) drift, regime-switching). See
`docs/problem.md` §"What this model is not good for".

### Documentation

Complete rewrite of `docs/model.md`, `docs/api.md`, `docs/design.md`,
`docs/problem.md`. New `docs/calibration.md` documenting every weight
choice. `docs/diagnostics.md` and `docs/workflow.md` updated for v0.5
expectations.

---

## [0.4.0] — May 2026 (developmental, never tagged)

The advective-SPDE iteration line. Several intermediate fits with
unsatisfactory diagnostic outputs led to the v0.5 stable release. Kept
only for changelog continuity.

### Intermediate findings during v0.4 development

- **v0.4 (advective + κ=0.01)**: Std ratio improved 0.12 → 0.16,
  directional 50% → 58%. Calibrated GARCH but the κ value broadened
  Gaussian enough for periodic-BC wrap-around at long horizons →
  midpoint-regression cliff.
- **v0.5 (advective + learnable σ_scale, weight 0.1, K=30)**:
  σ_scale = 4.0, ratio jumped to 0.29 then to 0.50 after cliff fix.
  But BFGS exploited high-persistence GARCH instead of σ_scale.
- **v0.6 (var-ratio weight 0.5)**: Forced clean GARCH (no unit-root)
  but σ_scale = 6.0 with K=30 rollout. Tail-event accumulation produced
  a cliff at day ~120.
- **v0.7 (κ = 1e-4)**: Smaller diffusion expected to fix cliff. Did
  not. Cliff persisted because root cause was sigma_scale × tail
  events, not boundary wrap-around.
- **v1.0 (K=30, K=150 in 1.1)**: K=150 rollout finally exposed
  tail-event contribution. σ_scale collapsed to 2.84, cliff gone,
  but predictions now under-amplified (std ratio 0.34).
- **v1.2 (K=150, shift weight 0.3)**: Goldilocks. σ_scale ≈ 4–6,
  std ratio ≈ 0.5–0.7, cliff gone, GARCH stays clean. → v0.5.0-stable.

---

## [0.3.0] — April 2026

### Added

- `stochax-merge`, `stochax-diagnose`, `stochax-visualize` CLIs.
- Center-of-mass `field_mean` formulation (mass-normalised) in
  `calibration/loss.py`.
- KL-noise eigenvalue anchoring: `λᵢ = (empirical_sigma / i)²` instead
  of the unit-amplitude default.

### Fixed (v0.2-line, pre-advective)

- **`predict.py` frozen-volatility bug.** Forecast σ trajectory was
  `jnp.full(horizon, last_sigma)` — held constant at the last
  historical σ. Fixed with a `lax.scan` of the GARCH recursion forward
  stochastically, so each MC path carries its own σ trajectory.
- **`predict.py` constant-drift bug.** Forecast drift was
  `jnp.full(horizon, last_drift)`, broadcasting the noisy single
  log-return on the final historical day across the entire horizon.
  Fixed by using the historical-mean drift.
- **Capitalisation mismatch.** V1.1.0 fetcher writes capitalised column
  names (Open, Close, etc.); downstream `validation.py` and
  `plotting.py` were reading lowercase. Fixed with case-insensitive
  `_resolve_column()` helper and explicit lowercasing where needed.

### Changed

- `stochax-fit` `--n-steps` flag is now honoured (no longer silently
  capped at 100).

---

## [0.2.0] — March 2026

### Added

- `stochax_market` package structure with `simulate`, `predict`,
  `fit` entry points.
- Initial SPDE model with multiplicative noise:
  `∂u/∂t = κ∇²u + μ·drift·u + σ·u·dW`.
- GARCH(1,1) `model/volatility.py` (no persistence cap yet).
- Optimistix BFGS calibration on bare MSE.

### Known issues at v0.2 (all resolved by v0.5)

- Calibration silently capped at 50 timesteps (band-aid for the
  single-trajectory rollout problem).
- `raw_omega = -3.0` default produced 13× inflated unconditional
  variance.
- Multiplicative noise couldn't move `field_mean` (root cause of std
  ratio 0.124 in May 2026 backtest).
- `simulate.py` forward drift was the historical log-return sequence
  played back as forecast.
- No persistence cap on GARCH → unit-root corner exploitable by any
  loss term that valued bursty volatility.
