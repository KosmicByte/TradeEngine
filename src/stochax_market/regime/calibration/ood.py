"""Out-of-distribution (novelty) scoring — the detector's "I don't know" signal.

A regime detector should flag when the current market state does not resemble anything it was
fit on, so downstream consumers can trust the label less. We summarise each detection as a
small fixed-length feature vector (drawn entirely from the ``RegimeResult`` — no recomputation)
and measure its Mahalanobis distance to the training distribution. The squared distance is
chi-square distributed under a Gaussian model, so we map it through the chi-square CDF to a
score in [0, 1]: near 0 = typical, near 1 = far in the tail = out-of-distribution.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

# Feature names define the vector order and dimensionality. Keep stable across fit/apply.
OOD_FEATURES = (
    "abs_lambda1",
    "arg_lambda1",
    "spectral_gap",
    "rank",
    "top_mode_energy",
    "realized_vol",
    "trend_strength",
)


def feature_vector(result: object) -> np.ndarray:
    """Build the OOD feature vector from a RegimeResult (duck-typed to avoid an import cycle).

    Pulls only fields the result already carries, so scoring adds no extra computation.
    """
    modes = result.modes  # type: ignore[attr-defined]
    eig = modes.eigenvalues
    if eig:
        lam1 = complex(eig[0])
        abs_lambda1 = abs(lam1)
        arg_lambda1 = abs(np.angle(lam1))
    else:
        abs_lambda1, arg_lambda1 = 1.0, 0.0
    top_energy = float(modes.mode_energies[0]) if modes.mode_energies else 0.0
    return np.array(
        [
            float(abs_lambda1),
            float(arg_lambda1),
            float(modes.spectral_gap),
            float(modes.rank),
            top_energy,
            float(result.realized_vol),  # type: ignore[attr-defined]
            float(result.trend_strength),  # type: ignore[attr-defined]
        ],
        dtype=float,
    )


def fit_reference(
    features: np.ndarray, ridge: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the OOD reference from a training feature matrix ``(n, d)``.

    Standardises features, then estimates a ridged inverse covariance of the standardised
    matrix (ridge keeps it invertible when features are collinear or n is small).

    Returns:
        ``(mean, std, inv_cov)`` — mean and std for standardisation, inv_cov for Mahalanobis.
    """
    features = np.atleast_2d(np.asarray(features, dtype=float))
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)  # guard constant features
    z = (features - mean) / std
    d = z.shape[1]
    cov = np.cov(z, rowvar=False) if len(z) > 1 else np.eye(d)
    cov = np.atleast_2d(cov) + ridge * np.eye(d)
    inv_cov = np.linalg.pinv(cov)
    return mean, std, inv_cov


def ood_score(
    x: np.ndarray, mean: np.ndarray, std: np.ndarray, inv_cov: np.ndarray
) -> float:
    """Map a feature vector to an OOD score in [0, 1] via the chi-square CDF of its distance.

    0 ≈ typical of training; values approaching 1 are increasingly atypical. A score of 0.95
    means the point is further from the centre than 95% of in-distribution points would be.

    Robust to non-finite inputs: a feature that could not be computed on a given window
    (NaN/inf — it happens on degenerate real-market windows) is imputed to the reference mean
    so it contributes neutrally to the distance rather than poisoning the score. The squared
    distance is clamped non-negative against pinv round-off, and any residual non-finite result
    falls back to 0.0 (treat-as-in-distribution) so a detection never returns a NaN score.
    """
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    # Impute non-finite features to the reference mean → standardised value 0 (neutral).
    bad = ~np.isfinite(x)
    if bad.any():
        x = x.copy()
        x[bad] = mean[bad]
    z = (x - mean) / std
    d2 = float(z @ inv_cov @ z)
    d2 = max(d2, 0.0)  # guard against tiny negative values from a pinv'd covariance
    if not np.isfinite(d2):
        return 0.0
    return float(chi2.cdf(d2, df=len(mean)))
