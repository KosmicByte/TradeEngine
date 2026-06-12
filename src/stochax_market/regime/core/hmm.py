"""Gaussian Hidden Markov Model on returns + rolling vol.

The classical regime-detection benchmark. We fit a k-state Gaussian HMM on a 2-D
feature stream (log return, log rolling vol), then label each state by its mean
return and mean vol — yielding interpretable states like 'high-vol-down',
'low-vol-up', etc.

This is our independent vote in the ensemble: a method whose statistical
assumptions are very different from the linear-operator view of EDMD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hmmlearn.hmm import GaussianHMM

from stochax_market.regime.core.observables import log_returns, rolling_std
from stochax_market.regime.models import RegimeLabel


@dataclass(frozen=True)
class HMMResult:
    """Output of an HMM fit + decode."""

    states: np.ndarray  # int array, Viterbi state per time
    state_means: np.ndarray  # (n_states, 2) — mean [return, log_vol] per state
    state_vols: np.ndarray  # (n_states,) — mean vol per state in original units
    transition_matrix: np.ndarray  # (n_states, n_states)
    posteriors: np.ndarray  # (T, n_states) — γ_t(i)
    state_labels: dict[int, RegimeLabel]  # which RegimeLabel each numeric state maps to
    current_label: RegimeLabel
    current_confidence: float


def _label_state(mean_ret: float, mean_vol: float, all_vols: np.ndarray) -> RegimeLabel:
    """Map (mean return, mean vol) of a state to our RegimeLabel taxonomy."""
    high_vol = mean_vol > np.median(all_vols) * 1.3
    low_vol = mean_vol < np.median(all_vols) * 0.7

    if high_vol and mean_ret < -0.0005:
        return RegimeLabel.TRENDING_DOWN
    if high_vol and mean_ret > 0.0005:
        return RegimeLabel.BREAKOUT
    if high_vol:
        return RegimeLabel.HIGH_VOL_CHOP
    if low_vol and abs(mean_ret) < 0.0003:
        return RegimeLabel.LOW_VOL_GRIND
    if mean_ret > 0.0005:
        return RegimeLabel.TRENDING_UP
    if mean_ret < -0.0005:
        return RegimeLabel.TRENDING_DOWN
    return RegimeLabel.MEAN_REVERTING


def fit_hmm(
    close: np.ndarray,
    n_states: int = 3,
    vol_window: int = 21,
    n_iter: int = 100,
    random_state: int = 42,
) -> HMMResult:
    """Fit a Gaussian HMM and produce a labelled regime sequence.

    Args:
        close: 1-D price series.
        n_states: number of latent regimes to fit.
        vol_window: rolling window for the vol feature.
        n_iter: EM iterations.
        random_state: RNG seed for reproducibility.
    """
    r = log_returns(close)
    if len(r) < vol_window + n_states * 5:
        raise ValueError(f"Need more data: have {len(r)} returns, need ~{vol_window + n_states*5}.")

    vol = rolling_std(r, vol_window)
    valid = ~np.isnan(vol)
    r_v = r[valid]
    vol_v = vol[valid]
    log_vol = np.log(np.maximum(vol_v, 1e-8))

    X = np.column_stack([r_v, log_vol])

    # Try "full" covariance first (most expressive); fall back to "diag" then "spherical"
    # if the data are too low-variance for a full covariance to remain positive-definite.
    # This happens routinely on low-vol-grind synthetic series.
    model: GaussianHMM | None = None
    last_exc: Exception | None = None
    for cov_type in ("full", "diag", "spherical"):
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=cov_type,
                n_iter=n_iter,
                random_state=random_state,
                min_covar=1e-5,  # regularises near-singular covariances
            )
            model.fit(X)
            # Smoke-test: predict will raise on singular covariance.
            _ = model.predict(X[:5])
            break
        except (ValueError, np.linalg.LinAlgError) as exc:
            last_exc = exc
            model = None
            continue
    if model is None:
        raise ValueError(f"HMM fit failed across all covariance types: {last_exc}")

    states = model.predict(X)
    posteriors = model.predict_proba(X)

    state_means = model.means_  # (n_states, 2)
    state_vols = np.exp(state_means[:, 1])

    state_labels = {
        i: _label_state(float(state_means[i, 0]), float(state_vols[i]), state_vols)
        for i in range(n_states)
    }

    current_state = int(states[-1])
    current_label = state_labels[current_state]
    current_confidence = float(posteriors[-1, current_state])

    return HMMResult(
        states=states,
        state_means=state_means,
        state_vols=state_vols,
        transition_matrix=model.transmat_,
        posteriors=posteriors,
        state_labels=state_labels,
        current_label=current_label,
        current_confidence=current_confidence,
    )
