# API Reference

## Data Layer

### `stochax_market.data.loader`

#### `load_stock(symbol: str) -> pd.DataFrame`

Download and return the CSV for a single NIFTY50 stock.

**Parameters:**
- `symbol` (str): Stock symbol (e.g. `'RELIANCE'`)

**Returns:** DataFrame sorted by Date ascending with columns: Date, Prev Close,
Open, High, Low, Last, Close, VWAP, etc.

**Example:**
```python
from stochax_market.data.loader import load_stock
df = load_stock("RELIANCE")
print(df.head())
```

#### `list_stocks() -> list[str]`

List available stock symbols in the NIFTY50 dataset.

**Returns:** List of symbol strings.

**Example:**
```python
from stochax_market.data.loader import list_stocks
symbols = list_stocks()
print(symbols[:5])
```

### `stochax_market.data.features`

#### `encode_features(df: pd.DataFrame, nx: int = 128) -> dict[str, jnp.ndarray]`

Encode NIFTY50 DataFrame columns into model-compatible JAX arrays.

**Parameters:**
- `df` (pd.DataFrame): Stock data with required columns
- `nx` (int): Number of spatial grid points (default: 128)

**Returns:** Dict with keys:
- `u0`: (T, nx) initial conditions
- `drift`: (T,) normalized drift
- `vwap_target`: (T,) normalized VWAP
- `close_target`: (T,) normalized Close
- `price_range`: (T, 2) [Low, High]
- `log_returns`: (T,) log-returns
- `dates`: (T,) integer day index
- `seasonal`: (T, 2) [sin, cos] seasonal features
- `price_scale`: float scaling factor

**Example:**
```python
from stochax_market.data.features import encode_features
features = encode_features(df, nx=64)
print(features["u0"].shape)
```

---

## Model Layer

### `stochax_market.model.spde.SPDEStepper`

Equinox module for SPDE time-stepping.

#### `__init__(kappa=0.01, mu_scale=0.1, nx=128, dt=1/252, domain_extent=1.0)`

**Parameters:**
- `kappa` (float): Diffusion coefficient
- `mu_scale` (float): Drift scaling factor
- `nx` (int): Spatial grid points
- `dt` (float): Time step size
- `domain_extent` (float): Spatial domain length L

#### `step(u, sigma_field, noise_increment, drift_scalar) -> jnp.ndarray`

One SPDE timestep.

**Parameters:**
- `u` (jnp.ndarray): Shape (nx,) current state
- `sigma_field` (jnp.ndarray): Shape (nx,) volatility field
- `noise_increment` (jnp.ndarray): Shape (nx,) noise sample
- `drift_scalar` (jnp.ndarray): Scalar drift value

**Returns:** Shape (nx,) updated state.

#### `rollout(u0, sigma_trajectory, noise_trajectory, drift_series, nt) -> jnp.ndarray`

Full trajectory via `jax.lax.scan`.

**Parameters:**
- `u0` (jnp.ndarray): Shape (nx,) initial condition
- `sigma_trajectory` (jnp.ndarray): Shape (nt, nx) volatility fields
- `noise_trajectory` (jnp.ndarray): Shape (nt, nx) noise increments
- `drift_series` (jnp.ndarray): Shape (nt,) drift values
- `nt` (int): Number of timesteps

**Returns:** Shape (nt, nx) trajectory.

**Example:**
```python
from stochax_market.model.spde import SPDEStepper
spde = SPDEStepper(kappa=0.01, nx=64)
trajectory = spde.rollout(u0, sigma_fields, noise, drift, 100)
```

### `stochax_market.model.volatility.GARCHVolatility`

Equinox module for GARCH(1,1) stochastic volatility.

#### `__init__(omega=0.01, alpha=0.05, beta=0.90)`

**Parameters:**
- `omega` (float): Base variance (ω > 0)
- `alpha` (float): ARCH coefficient (α ≥ 0)
- `beta` (float): GARCH coefficient (β ≥ 0)

#### `__call__(log_returns: jnp.ndarray) -> jnp.ndarray`

Compute GARCH(1,1) conditional volatilities.

**Parameters:**
- `log_returns` (jnp.ndarray): Shape (T,) log-returns

**Returns:** Shape (T,) volatilities σ_t.

#### `to_spatial_field(sigma_t, nx) -> jnp.ndarray` (static method)

Broadcast σ_t to spatial field of shape (nx,).

**Example:**
```python
from stochax_market.model.volatility import GARCHVolatility
garch = GARCHVolatility(omega=0.01, alpha=0.05, beta=0.90)
sigma_series = garch(log_returns)
```

### `stochax_market.model.noise`

#### `make_wiener_sample(key, nx, n_modes, eigenvalues, dx) -> jnp.ndarray`

Generate a single spatial Wiener sample via Karhunen-Loève expansion.

**Parameters:**
- `key`: JAX random key
- `nx` (int): Spatial grid points
- `n_modes` (int): Number of KL modes
- `eigenvalues` (jnp.ndarray): Shape (n_modes,) eigenvalues
- `dx` (float): Grid spacing

**Returns:** Shape (nx,) spatial noise sample.

#### `make_noise_trajectory(key, nt, nx, dt, n_modes=16, decay_rate=2.0, dx=None) -> jnp.ndarray`

Generate noise trajectory of shape (nt, nx).

**Returns:** Shape (nt, nx) noise increments.

**Example:**
```python
from stochax_market.model.noise import make_noise_trajectory
noise = make_noise_trajectory(jax.random.key(0), 100, 128, 1/252)
```

### `stochax_market.model.initial`

#### `price_to_field(price, L, nx, width=0.05) -> jnp.ndarray`

Create Gaussian blob initial condition.

**Parameters:**
- `price` (float): Center coordinate (normalized)
- `L` (float): Domain extent
- `nx` (int): Grid points
- `width` (float): Gaussian width

**Returns:** Shape (nx,) normalized field.

---

## Calibration Layer

### `stochax_market.calibration.loss`

#### `field_mean(u, x_grid) -> float`

Compute expected price: ∫ x·u(x) dx.

#### `calibration_loss(params, data_batch, model_fn, noise_key) -> float`

MSE + 0.1 × directional penalty.

### `stochax_market.calibration.fit`

#### `fit(model_spde, model_garch, data, n_steps=100, lr=1e-3, key=None) -> tuple`

Fit parameters using Optimistix BFGS.

**Returns:** `(fitted_spde, fitted_garch, loss_history_dict)`

**Example:**
```python
from stochax_market.calibration.fit import fit
fitted_spde, fitted_garch, info = fit(spde, garch, features, n_steps=50)
```

---

## Entrypoints

### `stochax_market.simulate`

#### `simulate(symbol, n_steps=252, output_path=None, seed=42, nx=128) -> dict`

Run SPDE forward simulation. Returns dict with predicted_prices, trajectory_shape, etc.

### `stochax_market.predict`

#### `predict(symbol, horizon=5, params_path=None, seed=42, n_samples=100, nx=128) -> dict`

Forecast prices with Monte Carlo confidence intervals. Returns dict with
mean_prediction, lower_ci, upper_ci, all_samples.

---

## CLI

### `stochax-simulate`

```bash
stochax-simulate --symbol RELIANCE --steps 252 --seed 42
```

### `stochax-fit`

```bash
stochax-fit --symbol RELIANCE --n-steps 500 --output params.pkl
```

### `stochax-predict`

```bash
stochax-predict --symbol RELIANCE --horizon 5 --params params.pkl
```
