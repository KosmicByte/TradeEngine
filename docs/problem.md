# Problem Statement

## Objective

Model the daily Close-price evolution of NIFTY50 equities with a
stochastic PDE that captures the essential statistical features of
financial time series — volatility clustering, fat tails, leverage —
and that produces calibrated **probability bands** suitable for risk
quantification.

The model is explicitly **a stochastic-process forecaster**, not a
regime predictor. See §"What this model is good for" below for the
honest scope.

## Why linear / ARMA models are insufficient

Classical linear time-series models (ARMA, ARIMA) assume:

- Constant conditional variance (homoscedasticity)
- Gaussian innovations
- Linear dependence structure

Financial returns violate all three:

1. **Non-constant variance** — the ACF of squared returns `|X_t|²` decays
   slowly, indicating long-range dependence in volatility. The
   homoscedasticity assumption fails.
2. **Fat-tailed innovations** — residuals from fitted ARMA models on
   RELIANCE 2014–2026 daily returns have excess kurtosis ≈ 8; the
   Gaussian assumption is empirically wrong.
3. **Nonlinear dynamics** — volatility responds asymmetrically to
   positive vs negative shocks (leverage effect), which is fundamentally
   nonlinear.

## Why GARCH alone is insufficient

GARCH(1,1) captures the volatility clustering but has limits:

1. **No price-distribution structure** — GARCH operates on a scalar
   series. There's no notion of a probability distribution over future
   prices, only a conditional variance.
2. **Multi-step forecast collapse** — multi-step GARCH variance forecasts
   converge geometrically to the unconditional variance, providing
   little information beyond a few steps.
3. **No spatial / cross-sectional structure** — extending to multi-asset
   or intraday spatial features requires a non-trivial framework lift.

## The SPDE approach

The advective SPDE wraps a GARCH process inside a spatial field:

```
∂u/∂t = κ ∇²u − μ_scale · drift · ∂u/∂x − σ_scale · σ(x) · ∂u/∂x · Ẇ
```

where:

- `u(x, t)` is a probability-like field on the normalised price domain
  `[0, 1]`.
- `σ(x, t)` is a spatial volatility field, with the per-time σ_t coming
  from a calibrated GARCH(1,1).
- The two advection terms (drift, noise) shift the centre of mass of
  `u` along the price axis. The Fokker–Planck-style structure means the
  centre of mass moves with the drift signal and random-walks with the
  noise.

The predicted price is the (mass-normalised) centre of mass:
`p̂(t) = L · ∫ x·u(x, t) dx / ∫ u(x, t) dx`.

This buys us:

1. **A probability distribution over future prices** — `u(x, t)` is the
   density, queryable for any quantile, not just the mean.
2. **GARCH-driven heteroscedasticity** with a coherent multi-step path
   distribution (each Monte Carlo path in `stochax-predict` has its own
   σ trajectory).
3. **Differentiable end-to-end** — `JAX + Equinox + Optimistix` allows
   calibration via gradient-based optimisation over 5 differentiable
   parameters.

The full math is in [`docs/model.md`](model.md).

## What this model is good for

Validated on RELIANCE 2014–2026 daily data:

- **Volatility forecasting and CI quantification.** Fitted GARCH(1,1)
  closely tracks 30-day realised vol (ρ ≈ 0.79). Unconditional GARCH
  variance matches realised variance within 5% post-calibration. The
  90% Monte Carlo CI bands from `stochax-predict` reflect proper
  multi-step rollout variance.
- **Risk and scenario quantification.** Use case: "where will the price
  spend 90% of its time over the next 21 days?" The model answers this
  with calibrated CI bands.
- **Daily-scale directional accuracy.** ~60–63% consecutive directional
  accuracy on the post-fit backtest window — better than chance but
  well below trade-relevant edge.

## What this model is not good for

- **Multi-month point prediction of trends in regime-shifted markets.**
  The drift term anchors to a scalar mean (full history or windowed).
  When the recent regime diverges sharply from the historical mean,
  the predicted central trajectory lags. Theil's U against a naive
  random walk lands in the 2.0–2.5 range on 150-day backtests for
  RELIANCE under the Apr 2026 regime shift (+15% historical drift vs
  −18% recent-60d).

  Mitigations: `--drift-window 60` or `126` on `stochax-simulate` /
  `stochax-predict` to lean into the recent regime. This isn't a bug
  — it's a structural property of the deterministic-drift channel.
  Improving it would require something like an AR(1) drift signal or
  regime-switching state; future-work territory.

- **Forecasting jumps or earnings surprises.** No jump-diffusion
  process; the SPDE is purely diffusive plus stochastic advection.

## Known structural limitations

1. **1D spatial domain.** Multi-asset correlations would require a
   multi-dimensional spatial field. Not currently supported.
2. **Periodic boundary conditions** (inherited from exponax). Safe at
   the current `κ = 1e-4` setting because the Gaussian initial condition
   barely broadens over a 150-day horizon (see `model.md` §3). Would
   require reflective BC if larger `κ` is needed.
3. **GARCH(1,1) only.** EGARCH or GJR-GARCH would model leverage
   effects more directly. Worth exploring if directional accuracy
   becomes a focus.
4. **Daily granularity.** No intraday signal; no overnight/intraday
   variance split.
5. **No jump processes.** Earnings, ex-dividends, and corporate-action
   discontinuities are not modelled.

## Honest version of the bottom line

`stochax-market` v0.5 is a well-calibrated stochastic-process model
that's good at variance and CI bands, mediocre at daily direction, and
weak at long-horizon point prediction in shifting regimes. It earns
its keep as a risk-quantification tool, not a directional forecaster.
The pipeline (`fit → simulate → merge → diagnose`) is the actual
deliverable; the diagnostic report tells you whether the model is
behaving as designed on any given run.
