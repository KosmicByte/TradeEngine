# Calibration Method — v0.5

This document explains the four-term penalised loss used in
`stochax-fit`, what each term identifies, and the empirical justification
for each weight value. If you're tweaking weights or adding a fifth
term, this is the place to understand what you're touching.

The full implementation lives in `src/stochax_market/calibration/fit.py`.

---

## Overview

The fit is an Optimistix BFGS minimisation over the 5-dimensional
parameter vector:

```
y = [raw_mu_scale, raw_sigma_scale, raw_omega, raw_alpha, raw_beta]
```

The loss is:

```
ℓ(y) = 1000 · MSE_TF
     + 0.1  · directional_TF
     + 0.5  · log²( σ²_unc(GARCH) / σ²_realised )
     + 0.3  · log²( var(rollout_diffs) / var(actual_diffs) )
```

- The first two terms (MSE and directional) are computed over **teacher-
  forced** one-step-ahead predictions at every timestep t ∈ [0, T).
  Each predicted price uses `u₀[t]` (the IC anchored at the previous
  day's close) as the starting state and runs **one** SPDE step.
- The third term anchors the unconditional GARCH variance to the
  realised log-return variance over the training window.
- The fourth term compares the variance of consecutive price diffs over
  a **free-running K=150-step rollout** to the variance of consecutive
  actual price diffs over the same 150-day window of training data.

The two regimes — teacher-forced single steps and free-running multi-day
rollouts — exist because they identify different things. MSE on
teacher-forced steps identifies the daily transition kernel; the rollout
shift variance identifies the long-horizon stochastic structure that a
forecaster actually produces. Without the second, BFGS would find a
local minimum where one-step predictions are well-tuned but free-running
rollouts produce smooth deterministic curves.

---

## Why each term has its current weight

### MSE — weight 1000

One-step-ahead MSE in normalised price space sits in the 1e-4 range at
a sensibly fit model (median absolute one-step error ≈ 0.004 normalised,
i.e. ≈ ₹6 for RELIANCE). A unit weight would make the MSE term
contribute ~1e-4 to the total loss, while the other terms naturally land
in the 0.01–0.5 range. With a unit weight, BFGS would be effectively
blind to MSE; raising it to 1000 brings the MSE-weighted term to ~0.1,
comparable to the others, so prediction quality actually drives the
gradient.

This was set in v0.6 (June 2026). Prior versions used unit weight on a
bare clipped MSE and BFGS made literally zero progress on `mu_scale`
because the gradient was buried under SPDE noise.

### Directional — weight 0.1

The directional term is the fraction of timesteps where
`sign(pred_diff) ≠ sign(target_diff)`. It's bounded `[0, 1]` and is
naturally on the same scale as the other components, so a small weight
(0.1) gives it a soft pressure without letting it dominate.

It rarely moves much during fit (typically lands ≈ 0.50, just below
chance), which is informative — it's telling you that the SPDE dynamics
mostly produce sign-agnostic random walks from a fresh-IC starting
point. Directional skill arrives from the drift channel, and the drift
channel is weakly identified at the daily one-step scale.

### Variance-ratio — weight 0.5

Anchors GARCH unconditional variance (`ω / (1 − α − β)`) to the realised
daily variance of log returns over the training window. The penalty is
`log²(unc / realised)`, which is zero when the ratio is 1, grows
symmetrically in log space, and is well-behaved at any starting point.

The weight was 0.01 in v0.6, was tripled to 0.5 in v0.7 after BFGS
exploited the cheap penalty. Specifically, at weight 0.01 the
variance-ratio term contributed at most ~0.07 to total loss; the
shift-variance penalty could be satisfied by pushing GARCH persistence
to ~0.999 (with ω compensating to keep the variance ratio nominally OK
but with massively bursty σ_t), and BFGS happily paid the 0.07 penalty.

At weight 0.5, the same persistence push would cost ~3.85 in
loss, which dominates everything else, forcing BFGS to use the
legitimate channel (raising `sigma_scale`) instead. This worked: in
v0.8+ fits, the variance ratio lands at 0.95–1.05 reliably.

### Shift-variance — weight 0.3

Compares the variance of consecutive predicted price diffs over a
K=150-step free-running rollout to the variance of consecutive actual
diffs over the same 150-day target window. This is what identifies
`sigma_scale`.

A teacher-forced MSE term alone would push `sigma_scale → 0` (any
uncorrelated noise raises one-step MSE; the minimum-MSE prediction is
just "no change"). The shift-variance term provides the opposing
pressure: more `sigma_scale` raises both predicted shift variance and
the rollout-trajectory variance ratio, and the penalty rewards matching
the target.

Weight evolution:
- **v0.6 (K=30 rollout, weight 0.1)**: `sigma_scale` landed at 4.0;
  rollout ratio 0.78; but BFGS exploited high GARCH persistence to
  contribute variance through volatility bursting rather than through
  `sigma_scale` — diagnosed and fixed in v0.7 via variance-ratio weight.
- **v0.7 (K=30, weight 0.1, var-ratio 0.5)**: `sigma_scale` = 6.0;
  rollout ratio 0.73; GARCH stays clean (no unit-root). However the
  K=30 rollout window is too short to penalise tail-event compounding —
  the 150-day forecast accumulates rare 3σ noise events into a "cliff
  dive" toward ₹900.
- **v1.1 (K=150, weight 0.1)**: `sigma_scale` collapsed to 2.84;
  rollout ratio dropped to 0.49; cliff disappeared but predictions are
  over-conservative.
- **v1.2 (K=150, weight 0.3)**: split the difference — `sigma_scale`
  ≈ 4–5, rollout ratio 0.5–0.7, no cliff, no over-correction.

The 0.3 weight gives BFGS enough pressure to land near a 0.7 rollout
ratio against current parameters without forcing it through GARCH-side
shortcuts.

### Why K=150

The K-step rollout window must be at least as long as the forecast
horizon the model is actually evaluated against. With K=30, BFGS sees
the variance of ±2σ daily moves but never sees the 3σ tail events that
compound over a 150-day rollout. The model then over-fits per-day
amplitude and produces unrealistic tail behaviour on the actual
forecast horizon — visible as a sudden 3-day collapse around step
120–125 in the v0.7 backtest.

K=150 makes the rollout span the same range of stochastic excursions
the forecast will face. Compute cost is 5× the K=30 rollout — a single
`lax.scan` of 150 sequential SPDE steps per loss evaluation — still
small compared to the T=3056 `vmap`'d teacher-forced steps that drive
MSE.

---

## Diagnostic preamble and postamble

`fit()` prints two diagnostic blocks (unless `--quiet`) so you can see
whether each component is doing real work.

### Preamble (before BFGS)

```
Calibration setup
  Mode                 : teacher-forced (MSE) + 150-step rollout (shift-var)
  Training window      : 3,056 / 3,056 timesteps (100.0%)
  Optimisation steps   : 1000 (BFGS cap)

Loss weights
  MSE                  : 1000.0
  Directional          : 0.1
  Variance-ratio       : 0.5
  Shift-variance       : 0.3

Initial parameters
  mu_scale (drift)     : 0.1000
  sigma_scale (noise)  : 1.0000
  ω, α, β              : 1.6702e-05, 0.1434, 0.8266
  Persistence (α+β)    : 0.9700
  Unconditional σ      : 0.0236 daily | 37.46% annualised
  Realised σ (window)  : 0.0168 daily | 26.74% annualised
  Unconditional / realised variance ratio : 1.96×
```

### Postamble (after BFGS)

```
Post-fit state
  mu_scale             : 0.1029
  sigma_scale          : 4.32
  ω, α, β              : 8.5e-06, 0.18, 0.79
  Persistence (α+β)    : 0.97
  Unconditional σ      : 0.0167 daily | 26.6% annualised

Rollout diff-std (final, 150-step window)
  Predicted diff std   : 0.0098
  Target    diff std   : 0.0127
  Ratio (pred/target)  : 0.77
```

If the postamble flags `⚠ Rollout under-amplified`, sigma_scale didn't
land high enough — raise `_SHIFT_VAR_WEIGHT`. If it flags `⚠ Rollout
over-amplified`, the opposite. The recommended range is roughly 0.5–1.2.

The postamble also shows the per-component loss decomposition with
weighted shares, so you can spot a term that's barely moving (if a
share is < 5% post-fit, that term may be near a constraint or have a
trivial minimum).

---

## When and what to retune

The defaults are tuned for daily NIFTY50 large-caps with ~12 years of
history. Common reasons to retune:

- **Per-instrument retuning**: small-caps with higher kurtosis may need
  a higher persistence cap (lift the constant in `volatility.py`); very
  liquid index-futures may want a lower cap. The diagnostics report's
  GARCH section flags if α+β ends up at the cap (which is also
  reportable from `result_info["garch_persistence"]`).

- **Shorter training windows**: passing `--training-window 252` or 504
  for a 1- or 2-year fit changes the realised-variance reference,
  which the variance-ratio term anchors to. The fit will adapt, but be
  aware that very short windows leave fewer training examples for the
  teacher-forced MSE term.

- **Different forecast horizons**: `K=150` is right when your forecast
  horizon is ~30–150 days. For >200-day forecasts, raise K
  proportionally (recompile cost: linear in K).

- **Per-asset-class thresholds**: the inline thresholds in
  `diagnostics.py` are equity-tuned. For commodities, FX, or rates,
  consider parameterising `analyze()` to accept a `thresholds` dict
  (see `docs/diagnostics.md` §Architecture and extension).

Any retuning should be tested with a backtest against actuals before
relying on it. The diagnostic report's `--out` markdown lets you
compare runs across `git diff`.

---

## Implementation references

| File                                | What's there                                        |
|-------------------------------------|------------------------------------------------------|
| `calibration/fit.py`                | `fit()`, `_run_model()`, all four loss components   |
| `calibration/loss.py`               | `field_mean()`, `calibration_loss()` (reference form) |
| `model/volatility.py`               | GARCH parameterisation and persistence cap          |
| `model/spde.py`                     | Advective step with mu/sigma scales                 |
| `cli.py::fit_main`                  | CLI surface for `--n-steps`, `--training-window`, `--quiet` |
