"""Optimistix-based parameter fitting for SPDE + GARCH model."""

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx

from stochax_market.calibration.loss import field_mean
from stochax_market.model.noise import make_noise_trajectory
from stochax_market.model.spde import SPDEStepper
from stochax_market.model.volatility import GARCHVolatility

_DEFAULT_BFGS_STEPS:     int   = 1000
_MAX_RESIDUAL_MAGNITUDE: float = 100.0

# ── Loss component weights ────────────────────────────────────────────────────
# Why these numbers
# -----------------
# MSE: weight 1000 to bring one-step MSE (~1e-4) into the 0.1 range so its
# gradient drives BFGS.
#
# Directional: small soft pressure toward sign-matching, unchanged.
#
# Variance-ratio: anchors GARCH unconditional variance to realised log-return
# variance.
#
# Shift-variance (ROLLOUT-BASED, v0.6 → K=150 in v1.1, June 2026):
#   v0.5: teacher-forced one-step shift variance.
#         Problem: TF over-predicts shift magnitude vs free-running rollout
#         (fresh-IC every step vs evolved-u). RELIANCE v0.5: TF ratio 0.86,
#         rollout ratio 0.29.
#   v0.6: K=30-step rollout shift variance vs actual diff variance.
#         Better, but 30-day window doesn't see tail events.
#   v1.1: K=150 rollout, matching the forecast horizon length so the
#         calibrated sigma_scale balances routine vs tail-event compounding.
#         See the _ROLLOUT_K docstring below for the cliff-dive forensics.
_MSE_WEIGHT:               float = 1000.0
_DIRECTIONAL_WEIGHT:       float = 0.1
_VARIANCE_RATIO_WEIGHT:     float = 0.5   # was 0.01
_SHIFT_VAR_WEIGHT:         float = 0.3   # bumped 0.1 → 0.3 in v1.2 (June 2026)
# v1.1 (K=150, weight=0.1) over-corrected: rollout ratio dropped from 0.78
# (v1.0 K=30) to 0.49, and the diagnostic std ratio dropped from 0.50 to 0.34.
# BFGS was content to leave shift-variance large because MSE-weighted (0.13)
# was comparable to shift-var-weighted (0.20). Weight 0.3 lifts shift-var-
# weighted to ~0.60 at the v1.1 minimum, forcing BFGS to push sigma_scale
# back up. Expected v1.2 outcome: sigma_scale ~4-6, rollout ratio ~0.7-0.9.

# Length of the rollout window used by the shift-variance penalty.
#
# Bumped from 30 → 150 in v1.1 (June 2026) after the v1.0 RELIANCE backtest
# revealed that K=30 was systematically under-penalising tail-event behaviour.
# The 30-day window samples ~30 daily diffs; with ~2% daily vol, the *largest*
# observed diff over 30 days is typically ~2σ. BFGS was thus fitting
# sigma_scale to match the variance of an ensemble dominated by ±2σ moves.
# In the 150-day forecast, however, ~5 days experience excursions of 3-4σ
# (independent KL-mode coincidences at the Gaussian's spatial location), and
# those tail events compound into the ~₹250 "cliff dive" observed in v1.0.
#
# K=150 lets BFGS see the same range of stochastic excursions the forecast
# will experience, so the calibrated sigma_scale will balance routine-day
# variance against tail-event compounding. Compute cost: 5× the K=30 rollout
# per loss eval (~150 sequential SPDE steps inside lax.scan); still
# negligible vs the T=3056 vmap'd one-step calls that drive MSE.
_ROLLOUT_K:                int   = 150


# ──────────────────────────────────────────────────────────────────────────────
# Forward model — teacher-forced one-step-ahead (used for MSE/directional)
# ──────────────────────────────────────────────────────────────────────────────

def _run_model(
    spde: SPDEStepper,
    garch: GARCHVolatility,
    data: dict[str, jnp.ndarray],
    noise_key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run the SPDE one step ahead at every timestep — teacher-forced.

    For each day t, the SPDE is stepped one day from ``u0_series[t]`` and
    the resulting field_mean is compared to ``close_target[t]``. Drives
    MSE/directional/var-ratio losses but NOT the shift-variance penalty
    (which now uses a rollout — see ``_rollout_shift_variance_penalty``).

    Returns (predicted_prices, sigma_series), both shape (T,).
    """
    log_returns = data["log_returns"]
    u0_series   = data["u0"]
    drift       = data["drift"]
    T  = log_returns.shape[0]
    nx = spde.nx

    sigma_series = garch(log_returns)
    sigma_fields = jax.vmap(GARCHVolatility.to_spatial_field, in_axes=(0, None))(
        sigma_series, nx
    )
    noise = make_noise_trajectory(noise_key, T, nx, spde.dt)

    def _one_step(u0_t, sigma_field_t, noise_t, drift_t):
        return spde.step(u0_t, sigma_field_t, noise_t, drift_t)

    u_predicted = jax.vmap(_one_step, in_axes=(0, 0, 0, 0))(
        u0_series, sigma_fields, noise, drift
    )

    x_grid    = jnp.linspace(0.0, spde.domain_extent, nx, dtype=jnp.float32)
    predicted = jax.vmap(field_mean, in_axes=(0, None))(u_predicted, x_grid)

    return predicted, sigma_series


# ──────────────────────────────────────────────────────────────────────────────
# Penalty components
# ──────────────────────────────────────────────────────────────────────────────

def _variance_ratio_penalty(
    garch: GARCHVolatility,
    realised_var: jnp.ndarray,
) -> jnp.ndarray:
    """log² of GARCH unconditional / realised variance ratio."""
    persistence = garch.alpha + garch.beta
    unc_var = garch.omega / jnp.maximum(1.0 - persistence, 1e-8)
    log_ratio = jnp.log(unc_var / (realised_var + 1e-12))
    return log_ratio ** 2


def _rollout_shift_variance_penalty(
    spde:           SPDEStepper,
    garch:          GARCHVolatility,
    data:           dict[str, jnp.ndarray],
    noise_key:      jax.Array,
    K:              int = _ROLLOUT_K,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Rollout-based shift-variance penalty.

    Runs a K-step free-running rollout from u0[T−K] using the GARCH-σ
    series, KL-Wiener noise and historical drift over the last K days.
    Compares the standard deviation of the resulting consecutive
    field-mean diffs to the standard deviation of actual close diffs
    over the same window. Returns log² of the variance ratio.

    Why this exists
    ~~~~~~~~~~~~~~~
    The v0.5 teacher-forced shift penalty calibrated against single-step
    shifts from a fresh, sharp Gaussian IC. Free-running rollouts produce
    systematically smaller per-step displacements (RELIANCE v0.5:
    TF ratio 0.86, rollout ratio 0.29 — a 3× gap). This penalty targets
    the rollout dynamics that simulate.py and the diagnostics report
    actually evaluate.

    Args:
        spde, garch, data: as in _run_model.
        noise_key: JAX key for the K-step rollout's noise.
        K: rollout window length.

    Returns:
        (penalty, pred_diff_std, target_diff_std) — penalty is a scalar
        log²(ratio); the std values are returned for diagnostic logging.
    """
    T_full = data["log_returns"].shape[0]
    K_eff  = min(int(K), int(T_full))
    nx     = spde.nx

    # Window: last K days of training data
    u0_start         = data["u0"][T_full - K_eff]
    log_returns_win  = data["log_returns"][-K_eff:]
    drift_win        = data["drift"][-K_eff:]
    target_win       = data["close_target"][-K_eff:]

    # Build σ + noise for the rollout window
    sigma_series_win = garch(log_returns_win)
    sigma_fields_win = jax.vmap(GARCHVolatility.to_spatial_field, in_axes=(0, None))(
        sigma_series_win, nx
    )
    noise_win = make_noise_trajectory(noise_key, K_eff, nx, spde.dt)

    # Free-running rollout (lax.scan inside spde.rollout)
    trajectory = spde.rollout(
        u0_start, sigma_fields_win, noise_win, drift_win, K_eff
    )

    x_grid     = jnp.linspace(0.0, spde.domain_extent, nx, dtype=jnp.float32)
    pred_prices = jax.vmap(field_mean, in_axes=(0, None))(trajectory, x_grid)

    pred_diffs   = jnp.diff(pred_prices)
    target_diffs = jnp.diff(target_win)

    pred_var   = jnp.var(pred_diffs)
    target_var = jnp.var(target_diffs)

    log_ratio = jnp.log((pred_var + 1e-12) / (target_var + 1e-12))
    return log_ratio ** 2, jnp.sqrt(pred_var), jnp.sqrt(target_var)


def _penalised_loss_parts(
    predicted: jnp.ndarray,
    target: jnp.ndarray,
    sigma_series: jnp.ndarray,
    spde: SPDEStepper,
    garch: GARCHVolatility,
    data: dict[str, jnp.ndarray],
    realised_var: jnp.ndarray,
    noise_key: jax.Array,
    max_residual: float = _MAX_RESIDUAL_MAGNITUDE,
) -> dict[str, jnp.ndarray]:
    """Decompose the calibration loss into its four components."""
    n = min(int(predicted.shape[0]), int(target.shape[0]))
    pred = predicted[:n]
    targ = target[:n]

    residuals = jnp.clip(pred - targ, -max_residual, max_residual)
    mse = jnp.mean(residuals ** 2)

    pred_diff = pred[1:] - pred[:-1]
    true_diff = targ[1:] - targ[:-1]
    sign_mismatch = jnp.mean(
        (jnp.sign(pred_diff) != jnp.sign(true_diff)).astype(jnp.float32)
    )

    var_ratio_penalty = _variance_ratio_penalty(garch, realised_var)
    shift_var_penalty, pred_diff_std, target_diff_std = (
        _rollout_shift_variance_penalty(spde, garch, data, noise_key)
    )

    return {
        "mse":               mse,
        "sign_mismatch":     sign_mismatch,
        "var_ratio_penalty": var_ratio_penalty,
        "shift_var_penalty": shift_var_penalty,
        "pred_diff_std":     pred_diff_std,
        "target_diff_std":   target_diff_std,
    }


def _penalised_loss(
    predicted: jnp.ndarray,
    target: jnp.ndarray,
    sigma_series: jnp.ndarray,
    spde: SPDEStepper,
    garch: GARCHVolatility,
    data: dict[str, jnp.ndarray],
    realised_var: jnp.ndarray,
    noise_key: jax.Array,
    max_residual: float = _MAX_RESIDUAL_MAGNITUDE,
) -> jnp.ndarray:
    """Scalar weighted penalised loss."""
    parts = _penalised_loss_parts(
        predicted, target, sigma_series,
        spde, garch, data, realised_var, noise_key, max_residual,
    )
    return (
        _MSE_WEIGHT             * parts["mse"]
        + _DIRECTIONAL_WEIGHT   * parts["sign_mismatch"]
        + _VARIANCE_RATIO_WEIGHT * parts["var_ratio_penalty"]
        + _SHIFT_VAR_WEIGHT     * parts["shift_var_penalty"]
    )


def _weighted_total_from_parts(parts: dict[str, jnp.ndarray]) -> float:
    return (
        _MSE_WEIGHT             * float(parts["mse"])
        + _DIRECTIONAL_WEIGHT   * float(parts["sign_mismatch"])
        + _VARIANCE_RATIO_WEIGHT * float(parts["var_ratio_penalty"])
        + _SHIFT_VAR_WEIGHT     * float(parts["shift_var_penalty"])
    )


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic preamble / postamble
# ──────────────────────────────────────────────────────────────────────────────

def _print_preamble(
    model_spde:  SPDEStepper,
    model_garch: GARCHVolatility,
    data:        dict[str, jnp.ndarray],
    T_full:      int,
    max_T:       int,
    n_steps:     int,
    init_parts:  dict[str, jnp.ndarray],
    init_pred:   jnp.ndarray,
    target:      jnp.ndarray,
    realised_sigma_d: float,
) -> None:
    """Pre-BFGS sanity block."""
    omega0 = float(model_garch.omega)
    alpha0 = float(model_garch.alpha)
    beta0  = float(model_garch.beta)
    persistence = alpha0 + beta0
    unc_sigma_d = float(math.sqrt(omega0 / max(1.0 - persistence, 1e-8)))
    unc_sigma_y = unc_sigma_d * math.sqrt(252.0)
    realised_sigma_y = realised_sigma_d * math.sqrt(252.0)
    var_ratio = (
        (unc_sigma_d / realised_sigma_d) ** 2 if realised_sigma_d > 0 else float("inf")
    )

    mean_drift_d = float(jnp.mean(data["drift"]))
    mean_drift_y = mean_drift_d * 252.0
    mu_scale    = float(model_spde.mu_scale)
    sigma_scale = float(model_spde.sigma_scale)

    n = min(int(init_pred.shape[0]), int(target.shape[0]))
    abs_err = jnp.abs(init_pred[:n] - target[:n])
    median_abs_err = float(jnp.median(abs_err))
    max_abs_err    = float(jnp.max(abs_err))

    pred_diff_std   = float(init_parts["pred_diff_std"])
    target_diff_std = float(init_parts["target_diff_std"])

    bar = "─" * 72
    print(bar)
    print("Calibration setup")
    print(f"  Mode                 : teacher-forced (MSE) + {_ROLLOUT_K}-step rollout (shift-var)")
    print(f"  Training window      : {max_T:,} / {T_full:,} timesteps "
          f"({max_T / max(T_full, 1):.1%})")
    print(f"  Optimisation steps   : {n_steps} (BFGS cap)")
    print()
    print("Loss weights")
    print(f"  MSE                  : {_MSE_WEIGHT}")
    print(f"  Directional          : {_DIRECTIONAL_WEIGHT}")
    print(f"  Variance-ratio       : {_VARIANCE_RATIO_WEIGHT}")
    print(f"  Shift-variance       : {_SHIFT_VAR_WEIGHT}   (rollout-based, K={_ROLLOUT_K})")
    print()
    print("Initial parameters")
    print(f"  mu_scale (drift)     : {mu_scale:.4f}")
    print(f"  sigma_scale (noise)  : {sigma_scale:.4f}")
    print(f"  ω, α, β              : {omega0:.4e}, {alpha0:.4f}, {beta0:.4f}")
    print(f"  Persistence (α+β)    : {persistence:.4f}")
    print(f"  Unconditional σ      : {unc_sigma_d:.4f} daily | {unc_sigma_y:.2%} annualised")
    print(f"  Realised σ (window)  : {realised_sigma_d:.4f} daily | {realised_sigma_y:.2%} annualised")
    print(f"  Unconditional / realised variance ratio : {var_ratio:.2f}×")
    if var_ratio > 10.0:
        print( "  ⚠ Unconditional variance ≫ realised — the variance-ratio penalty")
        print( "    will pull BFGS toward lower persistence / lower ω.")
    elif var_ratio < 0.1:
        print( "  ⚠ Unconditional variance ≪ realised — raise raw_omega init.")
    print()
    print("Drift (over training window)")
    print(f"  Mean daily drift     : {mean_drift_d:+.4e}")
    print(f"  Annualised drift     : {mean_drift_y:+.2%}")
    print()
    print("One-step prediction error (initial)")
    print(f"  Median |pred − target| : {median_abs_err:.6f}  (normalised price)")
    print(f"  Max    |pred − target| : {max_abs_err:.6f}")
    print()
    print(f"Rollout diff-std (initial, {_ROLLOUT_K}-step window)")
    print(f"  Predicted diff std   : {pred_diff_std:.6f}  (normalised)")
    print(f"  Target    diff std   : {target_diff_std:.6f}  (normalised)")
    if target_diff_std > 0:
        init_ratio = pred_diff_std / target_diff_std
        print(f"  Ratio (pred/target)  : {init_ratio:.3f}")
    print()
    init_total = _weighted_total_from_parts(init_parts)
    mse_w     = _MSE_WEIGHT             * float(init_parts['mse'])
    dir_w     = _DIRECTIONAL_WEIGHT     * float(init_parts['sign_mismatch'])
    var_w     = _VARIANCE_RATIO_WEIGHT  * float(init_parts['var_ratio_penalty'])
    shift_w   = _SHIFT_VAR_WEIGHT       * float(init_parts['shift_var_penalty'])
    print("Initial loss decomposition (raw | weighted | share)")
    print(f"  MSE                  : {float(init_parts['mse']):.6e}  | "
          f"{mse_w:.4e} | {mse_w / max(init_total, 1e-12):.1%}")
    print(f"  Directional penalty  : {float(init_parts['sign_mismatch']):.4f}      "
          f"  | {dir_w:.4e} | {dir_w / max(init_total, 1e-12):.1%}")
    print(f"  Variance-ratio       : {float(init_parts['var_ratio_penalty']):.4f}      "
          f"  | {var_w:.4e} | {var_w / max(init_total, 1e-12):.1%}")
    print(f"  Shift-variance       : {float(init_parts['shift_var_penalty']):.4f}      "
          f"  | {shift_w:.4e} | {shift_w / max(init_total, 1e-12):.1%}")
    print(f"  Initial total loss   : {init_total:.6e}")
    print(bar)


def _print_postamble(
    fitted_spde:  SPDEStepper,
    fitted_garch: GARCHVolatility,
    final_parts:  dict[str, jnp.ndarray],
    final_loss:   float,
    init_total:   float,
    bfgs_result:  str,
) -> None:
    """Post-BFGS summary."""
    omega_f = float(fitted_garch.omega)
    alpha_f = float(fitted_garch.alpha)
    beta_f  = float(fitted_garch.beta)
    persistence = alpha_f + beta_f
    unc_sigma_d = float(math.sqrt(omega_f / max(1.0 - persistence, 1e-8)))
    mu_scale_f    = float(fitted_spde.mu_scale)
    sigma_scale_f = float(fitted_spde.sigma_scale)

    pred_diff_std   = float(final_parts["pred_diff_std"])
    target_diff_std = float(final_parts["target_diff_std"])

    bar = "─" * 72
    print()
    print(bar)
    print("Post-fit state")
    print(f"  mu_scale             : {mu_scale_f:.4f}")
    print(f"  sigma_scale          : {sigma_scale_f:.4f}")
    print(f"  ω, α, β              : {omega_f:.4e}, {alpha_f:.4f}, {beta_f:.4f}")
    print(f"  Persistence (α+β)    : {persistence:.4f}")
    print(f"  Unconditional σ      : {unc_sigma_d:.4f} daily | "
          f"{unc_sigma_d * math.sqrt(252.0):.2%} annualised")
    print()
    print(f"Rollout diff-std (final, {_ROLLOUT_K}-step window)")
    print(f"  Predicted diff std   : {pred_diff_std:.6f}  (normalised)")
    print(f"  Target    diff std   : {target_diff_std:.6f}  (normalised)")
    if target_diff_std > 0:
        final_ratio = pred_diff_std / target_diff_std
        print(f"  Ratio (pred/target)  : {final_ratio:.3f}")
        if final_ratio < 0.5:
            print( "  ⚠ Rollout under-amplified — try raising _SHIFT_VAR_WEIGHT.")
        elif final_ratio > 2.0:
            print( "  ⚠ Rollout over-amplified — try lowering _SHIFT_VAR_WEIGHT.")
    print()
    mse_w   = _MSE_WEIGHT             * float(final_parts['mse'])
    dir_w   = _DIRECTIONAL_WEIGHT     * float(final_parts['sign_mismatch'])
    var_w   = _VARIANCE_RATIO_WEIGHT  * float(final_parts['var_ratio_penalty'])
    shift_w = _SHIFT_VAR_WEIGHT       * float(final_parts['shift_var_penalty'])
    print("Final loss decomposition (raw | weighted)")
    print(f"  MSE                  : {float(final_parts['mse']):.6e}  | {mse_w:.4e}")
    print(f"  Directional penalty  : {float(final_parts['sign_mismatch']):.4f}      "
          f"  | {dir_w:.4e}")
    print(f"  Variance-ratio       : {float(final_parts['var_ratio_penalty']):.4f}      "
          f"  | {var_w:.4e}")
    print(f"  Shift-variance       : {float(final_parts['shift_var_penalty']):.4f}      "
          f"  | {shift_w:.4e}")
    print(f"  Total                : {final_loss:.6e}")
    if init_total > 0:
        improvement = (init_total - final_loss) / init_total
        print(f"  Improvement vs init  : {improvement:+.2%}")
        if improvement < 1e-3:
            print( "  ⚠ BFGS made essentially no progress — loss landscape is flat "
                   "at init. Inspect gradient.")
    print(f"  BFGS result          : {bfgs_result}")
    print(bar)


# ──────────────────────────────────────────────────────────────────────────────
# Public fit() entry point
# ──────────────────────────────────────────────────────────────────────────────

def fit(
    model_spde: SPDEStepper,
    model_garch: GARCHVolatility,
    data: dict[str, jnp.ndarray],
    n_steps: int = _DEFAULT_BFGS_STEPS,
    training_window: int | None = None,
    lr: float = 1e-3,
    key: jax.Array | None = None,
    verbose: bool = True,
) -> tuple[SPDEStepper, GARCHVolatility, dict[str, Any]]:
    """Fit SPDE + GARCH parameters using Optimistix BFGS.

    Calibration setup (v0.6, June 2026)
    -----------------------------------
    Two parallel evaluations per loss call:

    1. **Teacher-forced one-step** for every t (vmap over T): drives MSE,
       directional and variance-ratio terms.
    2. **K-step free-running rollout** (lax.scan) from u0[T−K]: drives the
       shift-variance penalty.

    Loss
    ----
        loss = 1000·MSE_teacher_forced
             + 0.1·directional_teacher_forced
             + 0.01·log²(GARCH_unc_var / realised_var)
             + 0.1 ·log²(rollout_diff_var / actual_diff_var)

    The rollout-based shift term replaces the v0.5 teacher-forced shift
    term, which systematically under-calibrated sigma_scale because TF
    single-steps from fresh-IC over-shoot relative to rollout-from-evolved-u
    (RELIANCE v0.5: TF shift ratio 0.86, rollout shift ratio 0.29).

    BFGS parameter vector (5-dim, unchanged):
        y[0] = raw_mu_scale      (spde)
        y[1] = raw_sigma_scale   (spde)
        y[2] = raw_omega         (garch)
        y[3] = raw_alpha         (garch)
        y[4] = raw_beta          (garch)
    """
    if key is None:
        key = jax.random.key(0)

    T_full = int(data["log_returns"].shape[0])
    if training_window is None:
        max_T = T_full
    else:
        max_T = min(T_full, int(training_window))

    small_data = {
        k: v[:max_T] if hasattr(v, "shape") and len(v.shape) > 0 else v
        for k, v in data.items()
    }

    realised_var     = jnp.var(small_data["log_returns"])
    realised_sigma_d = float(jnp.sqrt(realised_var))

    init_pred, init_sigma = _run_model(model_spde, model_garch, small_data, key)
    init_parts = _penalised_loss_parts(
        init_pred, small_data["close_target"], init_sigma,
        model_spde, model_garch, small_data, realised_var, key,
    )
    init_total = _weighted_total_from_parts(init_parts)
    if verbose:
        _print_preamble(
            model_spde, model_garch, small_data,
            T_full=T_full, max_T=max_T, n_steps=n_steps,
            init_parts=init_parts,
            init_pred=init_pred,
            target=small_data["close_target"],
            realised_sigma_d=realised_sigma_d,
        )

    y0 = jnp.array([
        model_spde.raw_mu_scale,
        model_spde.raw_sigma_scale,
        model_garch.raw_omega,
        model_garch.raw_alpha,
        model_garch.raw_beta,
    ], dtype=jnp.float32)

    frozen_spde  = model_spde
    frozen_garch = model_garch

    def loss_fn(y, args):
        noise_key = args
        spde = eqx.tree_at(
            lambda s: (s.raw_mu_scale, s.raw_sigma_scale),
            frozen_spde,
            (y[0], y[1]),
        )
        garch = eqx.tree_at(
            lambda g: (g.raw_omega, g.raw_alpha, g.raw_beta),
            frozen_garch,
            (y[2], y[3], y[4]),
        )
        predicted, sigma_series = _run_model(spde, garch, small_data, noise_key)
        return _penalised_loss(
            predicted, small_data["close_target"], sigma_series,
            spde, garch, small_data, realised_var, noise_key,
        )

    solver = optx.BFGS(rtol=1e-5, atol=1e-5)

    try:
        sol = optx.minimise(
            loss_fn,
            solver,
            y0,
            args=key,
            max_steps=n_steps,
            throw=False,
        )
        y_opt      = sol.value
        final_loss = float(loss_fn(y_opt, key))

        fitted_spde = eqx.tree_at(
            lambda s: (s.raw_mu_scale, s.raw_sigma_scale),
            model_spde,
            (y_opt[0], y_opt[1]),
        )
        fitted_garch = eqx.tree_at(
            lambda g: (g.raw_omega, g.raw_alpha, g.raw_beta),
            model_garch,
            (y_opt[2], y_opt[3], y_opt[4]),
        )

        final_pred, final_sigma = _run_model(
            fitted_spde, fitted_garch, small_data, key
        )
        final_parts = _penalised_loss_parts(
            final_pred, small_data["close_target"], final_sigma,
            fitted_spde, fitted_garch, small_data, realised_var, key,
        )

        if verbose:
            _print_postamble(
                fitted_spde, fitted_garch,
                final_parts, final_loss, init_total, str(sol.result),
            )

        post_omega = float(fitted_garch.omega)
        post_persistence = float(fitted_garch.alpha + fitted_garch.beta)
        post_unc_var = post_omega / max(1.0 - post_persistence, 1e-8)
        post_var_ratio = post_unc_var / max(float(realised_var), 1e-12)

        final_pred_diff_std   = float(final_parts["pred_diff_std"])
        final_target_diff_std = float(final_parts["target_diff_std"])
        final_diff_std_ratio  = (
            final_pred_diff_std / final_target_diff_std
            if final_target_diff_std > 0 else float("inf")
        )

        result_info = {
            "n_steps":             n_steps,
            "training_window":     max_T,
            "mode":                "tf_msedir_plus_rollout_shiftvar_v1.2_K150_w0.3",
            "rollout_K":           _ROLLOUT_K,
            "loss_weights":        {
                "mse":               _MSE_WEIGHT,
                "directional":       _DIRECTIONAL_WEIGHT,
                "variance_ratio":    _VARIANCE_RATIO_WEIGHT,
                "shift_variance":    _SHIFT_VAR_WEIGHT,
            },
            "result":              str(sol.result),
            "initial_loss":        init_total,
            "final_loss":          final_loss,
            "final_mse":           float(final_parts["mse"]),
            "final_directional":   float(final_parts["sign_mismatch"]),
            "final_var_ratio_pen": float(final_parts["var_ratio_penalty"]),
            "final_shift_var_pen": float(final_parts["shift_var_penalty"]),
            "rollout_diff_std_pred":   final_pred_diff_std,
            "rollout_diff_std_target": final_target_diff_std,
            "rollout_diff_std_ratio":  final_diff_std_ratio,
            "garch_omega":         post_omega,
            "garch_alpha":         float(fitted_garch.alpha),
            "garch_beta":          float(fitted_garch.beta),
            "garch_persistence":   post_persistence,
            "unc_var_ratio":       post_var_ratio,
            "mu_scale":            float(fitted_spde.mu_scale),
            "sigma_scale":         float(fitted_spde.sigma_scale),
        }
    except Exception as e:
        fitted_spde  = model_spde
        fitted_garch = model_garch
        result_info  = {
            "n_steps":         0,
            "training_window": max_T,
            "mode":            "tf_msedir_plus_rollout_shiftvar_v0.6",
            "error":           str(e),
        }

    return fitted_spde, fitted_garch, result_info
