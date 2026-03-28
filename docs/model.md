# Mathematical Model: GARCH-Diffusion SPDE

## 1. From GBM to SPDE

### Geometric Brownian Motion (GBM)

The classical model for stock prices is:

```
dP_t = μ P_t dt + σ P_t dB_t
```

Under GBM, log-returns are:

```
X_t = log(P_{t+1}/P_t) = (μ - σ²/2)Δt + σ ΔB_t ~ N((μ - σ²/2)Δt, σ²Δt)
```

### Why Constant σ Fails

Empirical evidence shows that real financial returns exhibit:

1. **Fat tails** — Return distributions have excess kurtosis (> 3), meaning extreme
   events occur more frequently than predicted by a Gaussian model.

2. **Volatility clustering** — Large returns tend to be followed by large returns
   (of either sign). The conditional variance σ²_t is stochastic and auto-correlated.

3. **Leverage effect** — Negative returns tend to increase future volatility more
   than positive returns of the same magnitude.

4. **Long memory in |X_t|** — Although X_t is approximately uncorrelated, |X_t| and
   X²_t exhibit slowly decaying autocorrelations.

### GARCH(1,1) Motivation

The GARCH(1,1) model captures volatility clustering:

```
σ²_t = ω + α X²_{t-1} + β σ²_{t-1}
```

**Parameters:**
- ω > 0: Base variance (unconditional variance floor)
- α ≥ 0: ARCH coefficient (sensitivity to recent shocks)
- β ≥ 0: GARCH coefficient (persistence of volatility)

**Stationarity condition:** α + β < 1

**Unconditional variance:** σ² = ω / (1 - α - β)

## 2. SPDE Extension

### Stochastic PDE Formulation

We extend the model to a spatial-temporal field u(x, t):

```
∂u/∂t = κ ∇²u + μ(x,t) u + σ(x,t) u · Ẇ(x,t)
```

where:
- u(x, t): State field over 1D price-space grid x ∈ [0, L]
- κ: Diffusion coefficient (price mean-reversion strength)
- μ(x, t): Drift term from market features
- σ(x, t): GARCH-diffusion volatility field
- Ẇ(x, t): Q-Wiener process (space-time noise)

### Mild Solution Formulation

The mild solution is:

```
u(t) = e^{tΔ} u₀ + ∫₀ᵗ e^{(t-s)Δ} σ(s) dW(s)
```

Discretized via pseudo-spectral exponential time differencing (ETD) using exponax.

## 3. Karhunen-Loève Noise Discretization

The Q-Wiener process is discretized using a truncated Karhunen-Loève expansion:

```
W(x) = Σᵢ √λᵢ ξᵢ φᵢ(x)
```

where:
- ξᵢ ~ N(0, 1) are i.i.d. standard normal random variables
- φᵢ(x) = √(2/L) sin(iπx/L) are the eigenfunctions of the Laplacian on [0, L]
- λᵢ = i^{-p} are eigenvalues with power-law decay (p controls spatial regularity)

## 4. Feature-to-Model-Component Mapping

| Dataset Column | Model Component | Role |
|---------------|----------------|------|
| Prev Close | u₀(x) | Initial condition (Gaussian blob centered at prev close) |
| Open | drift offset | μ_scale · (Open - Prev Close) / Prev Close |
| High | spatial upper bound | Price range constraint |
| Low | spatial lower bound | Price range constraint |
| VWAP | calibration target | Weighted mean of spatial field |
| Close | terminal target | u(x, T) mean calibrates against Close |
| Last | secondary target | Cross-validation with Close |
| Date | time index | Seasonal features (day-of-week, month) |
