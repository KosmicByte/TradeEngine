"""Probability calibration via temperature scaling.

The ensemble's soft vote produces numbers that *look* like probabilities but are not
calibrated — when it says 0.70 it is not right 70% of the time. Temperature scaling fixes the
sharpness of the distribution with a single learned scalar T, fit by minimising negative
log-likelihood on a held-out set.

A deliberate, useful property: applying temperature scaling to a probability vector via
``softmax(log(p) / T)`` is monotonic in each coordinate's order, so **it never changes the
argmax label** — only how confident the model is. Calibration therefore cannot regress label
accuracy; it only makes the confidence honest. T > 1 softens an over-confident model (the
common case here); T < 1 sharpens an under-confident one; T = 1 is a no-op.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-12


def _to_logits(probs: np.ndarray) -> np.ndarray:
    """Treat log-probabilities as logits for temperature scaling."""
    return np.log(np.clip(probs, _EPS, 1.0))


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scale a probability matrix ``(n, K)`` (or vector ``(K,)``).

    Returns a re-normalised distribution of the same shape. The per-row argmax is preserved.
    """
    probs = np.atleast_2d(np.asarray(probs, dtype=float))
    z = _to_logits(probs) / max(float(temperature), _EPS)
    z -= z.max(axis=1, keepdims=True)  # stabilise
    e = np.exp(z)
    out = e / e.sum(axis=1, keepdims=True)
    return out[0] if out.shape[0] == 1 and np.ndim(probs) == 2 and probs.shape[0] == 1 else out


def fit_temperature(probs: np.ndarray, true_idx: np.ndarray) -> float:
    """Fit the temperature that minimises multiclass NLL on a calibration set.

    Args:
        probs: ``(n, K)`` raw probability rows.
        true_idx: ``(n,)`` integer index of the true label in each row.

    Returns:
        The optimal temperature in [0.05, 20]. Falls back to 1.0 on degenerate input.
    """
    probs = np.asarray(probs, dtype=float)
    true_idx = np.asarray(true_idx, dtype=int)
    if len(probs) == 0:
        return 1.0
    logits = _to_logits(probs)
    rows = np.arange(len(probs))

    def nll(temp: float) -> float:
        z = logits / max(temp, _EPS)
        z -= z.max(axis=1, keepdims=True)
        log_norm = np.log(np.exp(z).sum(axis=1))
        log_p_true = z[rows, true_idx] - log_norm
        return float(-np.mean(log_p_true))

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(res.x) if res.success else 1.0
