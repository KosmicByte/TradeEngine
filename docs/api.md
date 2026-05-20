# API Reference

Function-level signatures and return shapes for the public `stochax_market`
modules. For the why-and-when, see `model.md`, `calibration.md`, and
`workflow.md`.

---

## Data layer

### `stochax_market.data.loader`

#### `load_stock(symbol: str) -> pd.DataFrame`

Load historical OHLCV for a single NIFTY50 symbol from `./data/{symbol}.csv`.

**Parameters:**
- `symbol` (str): Ticker (e.g. `"RELIANCE"`).

**Returns:** `pd.DataFrame`, ascending by Date, with V1.1.0 capitalised
columns (Date, Symbol, Series, Prev Close, Open, High, Low, Close,
Volume, VWAP, Turnover, Trades, Deliverable Volume, %Deliverble).
Dates are tz-naive.

**Raises:** `FileNotFoundError` if `./data/{symbol}.csv` does not exist.

---

### `stochax_market.data.features`

#### `encode_features(df: pd.DataFrame, nx: int = 128) -> dict[str, ...]`

Build the feature dict consumed by `simulate`, `predict`, and `fit`.

**Parameters:**
- `df`: Output of `load_stock`.
- `nx` (int): Spatial grid points (default 128).

**Returns:** dict with keys:

| Key             | Shape    | Description                                                   |
|-----------------|----------|---------------------------------------------------------------|
| `u0`            | (T, nx)  | Gaussian initial conditions, one per day, centred at prev close |
| `drift`         | (T,)     | Per-day log-returns, used as drift signal in calibration      |
| `log_returns`   | (T,)     | Log returns (split-event-clipped at ±0.25)                    |
| `vwap_target`   | (T,)     | Normalised VWAP                                              |
| `close_target`  | (T,)     | Normalised Close                                             |
| `price_range`   | (T, 2)   | Normalised [Low, High]                                       |
| `dates`         | (T,)     | NumPy datetime array                                         |
| `seasonal`      | (T, 2)   | (sin, cos) of day-of-year                                    |
| `L`             | float    | `max(High)` over the dataset; the price scale                |
| `price_scale`   | float    | Equal to `L` (backward-compat alias)                         |

Normalisation: all price columns are divided by `L`, so the SPDE works
on `[0, 1]`. To recover INR prices, multiply by `L`.

---

## Model layer

### `stochax_market.model.spde.SPDEStepper`

Equinox module implementing the advective SPDE step.

#### `__init__(kappa=1e-4, mu_scale=0.1, sigma_scale=1.0, nx=128, dt=1/252, domain_extent=1.0)`

**Parameters:**
- `kappa` (float): Diffusion coefficient (concrete value for exponax).
  Default `1e-4`; do not raise without reading the wrap-around note in
  `model.md` §3.
- `mu_scale` (float): Initial drift advection scale. Learnable
  (softplus-constrained > 0); BFGS updates it.
- `sigma_scale` (float): Initial noise advection scale. Learnable
  (softplus-constrained > 0); BFGS updates it.
- `nx` (int): Spatial grid points.
- `dt` (float): Time step size.
- `domain_extent` (float): Spatial domain length L.

**Properties:**
- `mu_scale: jnp.ndarray` — `softplus(raw_mu_scale)`, the effective
  drift scale.
- `sigma_scale: jnp.ndarray` — `softplus(raw_sigma_scale)`, the
  effective noise scale.

#### `step(u, sigma_field, noise_increment, drift_scalar) -> jnp.ndarray`

One SPDE timestep.

**Parameters:**
- `u` (jnp.ndarray): Shape `(nx,)`, current field.
- `sigma_field` (jnp.ndarray): Shape `(nx,)`, spatial volatility field
  for this step. Use `GARCHVolatility.to_spatial_field(σ_t, nx)`.
- `noise_increment` (jnp.ndarray): Shape `(nx,)`, the per-step KL-Wiener
  noise increment `√dt · W(x)`.
- `drift_scalar` (jnp.ndarray): Scalar, the per-step drift.

**Returns:** Shape `(nx,)`, the updated field.

The step computes
`u_diff − μ_scale·drift·∂u/∂x·dt − σ_scale·σ(x)·∂u/∂x·dW`, with the
diffusion applied first via exponax and the two advection terms added
in physical space. Final clip at `max(0, ·)` for non-negativity.

#### `rollout(u0, sigma_trajectory, noise_trajectory, drift_series, nt) -> jnp.ndarray`

Full forward trajectory via `jax.lax.scan`.

**Parameters:**
- `u0` (jnp.ndarray): Shape `(nx,)`, initial condition.
- `sigma_trajectory` (jnp.ndarray): Shape `(nt, nx)`.
- `noise_trajectory` (jnp.ndarray): Shape `(nt, nx)`.
- `drift_series` (jnp.ndarray): Shape `(nt,)`.
- `nt` (int): Number of timesteps.

**Returns:** Shape `(nt, nx)`, the trajectory of `u(·, t)` at each step.

---

### `stochax_market.model.volatility.GARCHVolatility`

GARCH(1,1) Equinox module with hard persistence cap.

#### `__init__(raw_omega=-11.0, raw_alpha=0.5, raw_beta=2.0)`

Parameters live in **unconstrained** space and are transformed at
access time. The defaults give `ω ≈ 1.7e-5`, `α ≈ 0.143`, β at the
persistence cap of 0.97 − α ≈ 0.827. Sensible starting point for daily
equity returns.

**Properties:**
- `omega: jnp.ndarray` — `softplus(raw_omega)` (positive)
- `alpha: jnp.ndarray` — `0.05 + 0.15 · sigmoid(raw_alpha)`, in [0.05, 0.20]
- `beta: jnp.ndarray` — `clip(0.50 + 0.40·sigmoid(raw_beta), max=0.97−α)`,
  effectively in [0.50, 0.92]

The hard cap `α + β ≤ 0.97` prevents the unit-root pathology described
in `model.md` §2.

#### `__call__(log_returns: jnp.ndarray) -> jnp.ndarray`

Compute the conditional volatility trajectory.

**Parameters:**
- `log_returns` (jnp.ndarray): Shape `(T,)`.

**Returns:** Shape `(T,)`, the per-day conditional volatility σ_t (daily,
log-return scale).

NaN log-returns (split-event days) are replaced by 0 in the recursion.

#### `to_spatial_field(sigma_t, nx) -> jnp.ndarray` (static)

Broadcast a scalar σ_t to a spatial field.

**Returns:** Shape `(nx,)`, `σ_t · nx · (1 + 0.1·sin(πx))`.

---

### `stochax_market.model.noise`

#### `make_wiener_sample(key, nx, n_modes, eigenvalues, dx) -> jnp.ndarray`

Generate one spatial Wiener sample via Karhunen-Loève expansion.

**Returns:** Shape `(nx,)`.

#### `make_noise_trajectory(key, nt, nx, dt, n_modes=32, decay_rate=2.0, dx=None, empirical_sigma=0.015) -> jnp.ndarray`

Generate `nt` independent noise samples scaled by `√dt`.

**Returns:** Shape `(nt, nx)`.

Eigenvalues are `λᵢ = (empirical_sigma / i)^decay_rate`, anchored to the
empirical daily volatility scale.

---

### `stochax_market.model.initial`

#### `price_to_field(price, L, nx, width=0.05) -> jnp.ndarray`

Gaussian blob initial condition centred at `price` on `[0, L]`.

#### `make_initial_condition(last_price, domain_extent, nx, sigma_width=50.0) -> jnp.ndarray`

Convert a raw INR price to a Gaussian field on the normalised `[0, 1]`
grid. `sigma_width` is the Gaussian width in raw price units (default
50 INR, normalised to `50/L` on the grid).

---

## Calibration layer

### `stochax_market.calibration.loss`

#### `field_mean(u, x_grid) -> jnp.ndarray`

Centre-of-mass price extraction from a field `u`.

**Formula:** `∫ x·u(x) dx / (∫ u(x) dx + ε)`. Mass normalisation
handles the small mass loss from `max(u, 0)` clipping in `SPDEStepper.step`.

**Returns:** Scalar.

#### `calibration_loss(params, data_batch, model_fn, noise_key) -> jnp.ndarray`

Reference penalised loss using the dict/model_fn API (used in tests and
exploration). For the Equinox-API form actually used in BFGS, see
`fit.py::_penalised_loss`.

---

### `stochax_market.calibration.fit`

#### `fit(model_spde, model_garch, data, n_steps=1000, training_window=None, lr=1e-3, key=None, verbose=True) -> tuple[SPDEStepper, GARCHVolatility, dict]`

Fit SPDE + GARCH via Optimistix BFGS on the 4-component penalised loss.

**Parameters:**
- `model_spde`: Initial `SPDEStepper`.
- `model_garch`: Initial `GARCHVolatility`.
- `data`: Output of `encode_features`.
- `n_steps` (int): Max BFGS iterations (default 1000).
- `training_window` (int | None): Restrict to last N days (default: full).
- `lr` (float): Unused (BFGS line-searches); kept for API stability.
- `key` (jax.Array | None): Random key (default `key(0)`).
- `verbose` (bool): Print diagnostic preamble and postamble.

**Returns:** Tuple of:
- `fitted_spde: SPDEStepper`
- `fitted_garch: GARCHVolatility`
- `result_info: dict` with keys:
  - `n_steps`, `training_window`, `mode`, `result` (str), `initial_loss`,
    `final_loss`
  - `final_mse`, `final_directional`, `final_var_ratio_pen`, `final_shift_var_pen`
  - `rollout_diff_std_pred`, `rollout_diff_std_target`, `rollout_diff_std_ratio`
  - `garch_omega`, `garch_alpha`, `garch_beta`, `garch_persistence`, `unc_var_ratio`
  - `mu_scale`, `sigma_scale`
  - `loss_weights: dict`

`diagnostics.py` consumes `result_info` for the Calibration Convergence
section.

---

## Entry points

### `stochax_market.simulate.simulate`

```python
simulate(symbol, n_steps=252, output_path=None, params_path=None,
         seed=42, nx=128, drift_window=None) -> dict
```

Single deterministic-seed forward rollout. Writes a CSV with `step` and
`predicted_price` columns; returns a dict with the predicted-price
array and metadata.

The forward drift is a **scalar mean**: full historical mean by
default, or the last `drift_window` days' mean if specified.

### `stochax_market.predict.predict`

```python
predict(symbol, horizon=5, params_path=None, seed=42, n_samples=100,
        nx=128, drift_window=None) -> dict
```

Monte Carlo forecast with `jax.vmap`. Returns:

| Key                | Shape              | Description                          |
|--------------------|--------------------|--------------------------------------|
| `mean_prediction`  | `(horizon,)`       | Mean across MC paths, in INR         |
| `lower_ci`         | `(horizon,)`       | 5th percentile                       |
| `upper_ci`         | `(horizon,)`       | 95th percentile                      |
| `all_samples`      | `(n_samples, horizon)` | Full ensemble                    |
| `horizon`          | int                | Echo of input                        |
| `n_samples`        | int                | Echo of input                        |

Internally rolls GARCH(1,1) forward stochastically (each MC path gets
its own σ trajectory) and uses windowed-mean drift consistent with
`simulate`.

### `stochax_market.merge.merge_predicted_with_actuals`

```python
merge_predicted_with_actuals(sim_csv, symbol, start_date=None, out=None) -> Path
```

Joins a `simulate` CSV with realised actuals from V1.1.0 historical
data. Adds `actual_price` and `date` columns. Idempotent.

### `stochax_market.diagnostics.analyze`

```python
analyze(symbol, sim_csv, params_path=None, out_md=None) -> DiagnosticsReport
```

Runs the seven diagnostic sections. Returns a `DiagnosticsReport` (see
`docs/diagnostics.md` for the dataclass shape). Optionally writes a
Markdown report to `out_md`.

### `stochax_market.visualize.run_all`

```python
run_all(symbol, sim_csv=None, params_path=None, horizon=21,
        recent_n=60, out_dir="plots", seed=42) -> dict[str, Path]
```

Generates the seven Plotly diagnostic plots depending on which inputs
are provided. Returns a dict mapping plot-name → saved path.

---

## CLI

All CLI commands wrap the corresponding entry-point function and add
Rich-formatted result tables. See `README.md` §Quick start for the
full surface, and `workflow.md` for end-to-end recipes.

| Command             | Key flags                                                          |
|---------------------|--------------------------------------------------------------------|
| `stochax-fit`       | `--n-steps 1000`, `--training-window N`, `--quiet`                 |
| `stochax-simulate`  | `--steps`, `--params`, `--seed`, `--drift-window N`                |
| `stochax-predict`   | `--horizon`, `--samples`, `--drift-window N`                       |
| `stochax-merge`     | `--start-date DD/MM/YY`                                            |
| `stochax-diagnose`  | `--params`, `--out`                                                |
| `stochax-visualize` | `--sim-csv`, `--params`, `--horizon`, `--recent-n`, `--out-dir`    |
| `stochax-export`    | `--params`, `--output`                                             |
