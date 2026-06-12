"""Fit the calibration artifact by harvesting detections over the synthetic battery.

This is the training step behind ``regime calibrate``. It runs the detector (with calibration
*off*, to collect raw probabilities) across walk-forward windows of the synthetic battery,
then fits temperature, the conformal threshold, and the OOD reference from that harvest.

The fit source is recorded as ``"synthetic"`` in the artifact, and the notes field carries the
caveat that coverage guarantees are indicative until re-fit on real data. When real NSE data
is wired in, this same routine takes a real-data harvest and the source becomes ``"nse_real"``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stochax_market.regime.calibration.artifact import CalibrationArtifact
from stochax_market.regime.calibration.calibrator import apply_temperature, fit_temperature
from stochax_market.regime.calibration.conformal import fit_threshold
from stochax_market.regime.calibration.ood import feature_vector, fit_reference, ood_score
from stochax_market.regime.core.contract import PointInTimeFrame
from stochax_market.regime.core.regime import detect_regime
from stochax_market.regime.models import RegimeLabel
from stochax_market.regime.version import MODEL_VERSION

# Canonical label order — the index convention for every probability vector and the artifact.
CANONICAL_LABELS: list[RegimeLabel] = list(RegimeLabel)
LABEL_NAMES: list[str] = [lbl.value for lbl in CANONICAL_LABELS]
_LABEL_INDEX = {lbl: i for i, lbl in enumerate(CANONICAL_LABELS)}

OOD_THRESHOLD = 0.99  # score above which a window is flagged out-of-distribution


@dataclass
class Harvest:
    """Raw detections collected for fitting."""

    probs: np.ndarray  # (n, K) raw soft-vote probabilities
    true_idx: np.ndarray  # (n,) true-label indices into CANONICAL_LABELS
    features: np.ndarray  # (n, d) OOD feature vectors


def harvest(
    series_battery,
    window: int = 252,
    step: int = 21,
    edmd_rank: int = 10,
    hmm_n_states: int = 3,
) -> Harvest:
    """Slide windows over each synthetic series, collecting raw probs, true labels, features.

    Detection runs with ``calibrate=False`` so we capture the *uncalibrated* distribution the
    calibrator is meant to correct.
    """
    rows_p: list[np.ndarray] = []
    rows_y: list[int] = []
    rows_f: list[np.ndarray] = []
    for s in series_battery:
        prices = s.prices
        labels = s.labels
        n_total = len(prices) - 1
        for end in range(window, n_total + 1, step):
            window_close = prices[end - window : end + 1]
            try:
                frame = PointInTimeFrame.from_arrays(window_close, symbol="synthetic")
                result = detect_regime(
                    frame=frame,
                    edmd_rank=min(edmd_rank, window // 3),
                    hmm_n_states=hmm_n_states,
                    calibrate=False,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            rows_p.append(np.array([result.probabilities[lbl] for lbl in CANONICAL_LABELS]))
            rows_y.append(_LABEL_INDEX[labels[end - 1]])
            rows_f.append(feature_vector(result))
    if not rows_p:
        raise RuntimeError("Harvest produced no windows — check the battery and window size.")
    return Harvest(
        probs=np.vstack(rows_p),
        true_idx=np.array(rows_y, dtype=int),
        features=np.vstack(rows_f),
    )


def fit_calibration(
    series_battery=None,
    *,
    window: int = 252,
    step: int = 21,
    edmd_rank: int = 10,
    hmm_n_states: int = 3,
    alpha: float = 0.1,
    fit_source: str = "synthetic",
) -> tuple[CalibrationArtifact, dict]:
    """Harvest and fit a full calibration artifact.

    Returns:
        ``(artifact, diagnostics)`` where diagnostics carries before/after calibration quality
        for the CLI to print (ECE, Brier, empirical coverage, harvested sample count).
    """
    if series_battery is None:
        from stochax_market.regime.eval.metrics import default_battery

        series_battery = default_battery()

    h = harvest(
        series_battery,
        window=window,
        step=step,
        edmd_rank=edmd_rank,
        hmm_n_states=hmm_n_states,
    )

    temperature = fit_temperature(h.probs, h.true_idx)
    calibrated = apply_temperature(h.probs, temperature)
    threshold = fit_threshold(calibrated, h.true_idx, alpha=alpha)
    mean, std, inv_cov = fit_reference(h.features)

    # Diagnostics (imported lazily to avoid a hard dependency from the fit path).
    from stochax_market.regime.eval.calibration_metrics import (
        brier_score,
        empirical_coverage,
        expected_calibration_error,
    )

    in_scores = np.array([ood_score(f, mean, std, inv_cov) for f in h.features])
    diagnostics = {
        "n_samples": int(len(h.probs)),
        "temperature": float(temperature),
        "ece_raw": expected_calibration_error(h.probs, h.true_idx),
        "ece_calibrated": expected_calibration_error(calibrated, h.true_idx),
        "brier_raw": brier_score(h.probs, h.true_idx),
        "brier_calibrated": brier_score(calibrated, h.true_idx),
        "coverage": empirical_coverage(calibrated, h.true_idx, threshold),
        "coverage_target": 1.0 - alpha,
        "ood_flag_rate": float(np.mean(in_scores > OOD_THRESHOLD)),
    }

    artifact = CalibrationArtifact(
        labels=LABEL_NAMES,
        temperature=float(temperature),
        conformal_threshold=float(threshold),
        coverage_level=float(1.0 - alpha),
        ood_mean=mean.tolist(),
        ood_std=std.tolist(),
        ood_inv_cov=inv_cov.tolist(),
        ood_dim=int(len(mean)),
        ood_threshold=OOD_THRESHOLD,
        fit_source=fit_source,
        model_version=MODEL_VERSION,
        notes=(
            "Fit on the synthetic battery. Coverage and OOD thresholds are indicative, not "
            "real-market guarantees; re-run `regime calibrate` on real data once it is wired in."
            if fit_source == "synthetic"
            else ""
        ),
    )
    return artifact, diagnostics
