"""Split-conformal prediction sets for regime classification.

A point label hides how sure the model is about *which* alternatives are plausible. Conformal
prediction turns the calibrated probabilities into a *set* of labels with a distribution-free
coverage guarantee: if the threshold targets 90% coverage, the true regime falls inside the
returned set at least ~90% of the time on exchangeable data — regardless of whether the
underlying probabilities are perfectly calibrated.

We use the standard split-conformal recipe with nonconformity score ``s = 1 - p(true)``:

    1. On a held-out calibration set, score every example by ``1 - calibrated_prob(true_label)``.
    2. Take the ``ceil((n + 1)(1 - alpha)) / n`` empirical quantile ``q`` of those scores.
    3. At test time the prediction set is ``{label : calibrated_prob(label) >= 1 - q}``.

Empty sets are avoided by always retaining the argmax, so the set is never less informative
than the point label.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def fit_threshold(cal_probs: np.ndarray, true_idx: np.ndarray, alpha: float = 0.1) -> float:
    """Fit the conformal threshold ``q`` for a target coverage of ``1 - alpha``.

    Args:
        cal_probs: ``(n, K)`` *calibrated* probability rows on the calibration set.
        true_idx: ``(n,)`` true-label indices.
        alpha: miscoverage rate; coverage target is ``1 - alpha`` (default 0.1 → 90%).

    Returns:
        Threshold ``q`` in [0, 1]. The prediction set is ``{k : prob_k >= 1 - q}``.
    """
    cal_probs = np.asarray(cal_probs, dtype=float)
    true_idx = np.asarray(true_idx, dtype=int)
    n = len(cal_probs)
    if n == 0:
        return 1.0  # degenerate → include everything (vacuously covered)
    scores = 1.0 - cal_probs[np.arange(n), true_idx]
    # Finite-sample-valid quantile level.
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def prediction_set(probs: np.ndarray, threshold: float, labels: list[str]) -> list[str]:
    """Return the conformal prediction set for one probability vector.

    Includes every label whose probability is at least ``1 - threshold``; always retains the
    argmax so the set is never empty. Ordered by descending probability.
    """
    probs = np.asarray(probs, dtype=float)
    cutoff = 1.0 - threshold
    keep = np.where(probs >= cutoff - _EPS)[0]
    if len(keep) == 0:
        keep = np.array([int(np.argmax(probs))])
    order = keep[np.argsort(-probs[keep])]
    return [labels[i] for i in order]
