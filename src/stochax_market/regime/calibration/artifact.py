"""The persisted calibration artifact — a single versioned bundle.

A :class:`CalibrationArtifact` packages everything detection needs to turn raw soft-vote
probabilities into calibrated, uncertainty-aware output: the temperature, the conformal
threshold, and the out-of-distribution reference. It is stored as plain JSON (numpy arrays
as nested lists) so it is inspectable with any tool and carries no pickle / version-lock risk.

Crucially it records *what it was fit on* (``fit_source``) and *with which detection logic*
(``model_version``). Today that source is ``"synthetic"`` — the calibrator is fit on the
synthetic battery because real NSE data is not yet wired in. The artifact is explicit about
this so nobody mistakes synthetic-fit coverage guarantees for real-market ones; re-fitting on
real data later is a single ``regime calibrate`` run, not a code change.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ARTIFACT_SCHEMA = 1


@dataclass
class CalibrationArtifact:
    """Everything required to calibrate a detection, plus provenance about the fit.

    Attributes:
        labels: ordered label names defining the index convention for all vectors/matrices.
        temperature: scalar T for temperature scaling of the probability vector.
        conformal_threshold: q such that the prediction set is {label : prob >= 1 - q}.
        coverage_level: the 1 - alpha the conformal threshold targets (e.g. 0.9).
        ood_mean / ood_std: per-feature standardisation of the OOD feature vector.
        ood_inv_cov: inverse covariance (of standardised features) for the Mahalanobis distance.
        ood_dim: feature dimensionality (= chi-square dof for the score).
        ood_threshold: score above which a window is flagged out-of-distribution.
        fit_source: what the artifact was fit on — "synthetic" today, "nse_real" once wired.
        model_version: detection-logic version the fit corresponds to.
        created_at: UTC ISO timestamp.
        notes: free-text caveats (e.g. the synthetic-fit warning).
    """

    labels: list[str]
    temperature: float
    conformal_threshold: float
    coverage_level: float
    ood_mean: list[float]
    ood_std: list[float]
    ood_inv_cov: list[list[float]]
    ood_dim: int
    ood_threshold: float
    fit_source: str
    model_version: str
    schema: int = ARTIFACT_SCHEMA
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""

    # ------------------------------------------------------------------ convenience
    @property
    def ood_mean_arr(self) -> np.ndarray:
        return np.asarray(self.ood_mean, dtype=float)

    @property
    def ood_std_arr(self) -> np.ndarray:
        return np.asarray(self.ood_std, dtype=float)

    @property
    def ood_inv_cov_arr(self) -> np.ndarray:
        return np.asarray(self.ood_inv_cov, dtype=float)

    def is_synthetic_fit(self) -> bool:
        """True when this artifact was fit on synthetic data (coverage is indicative only)."""
        return self.fit_source == "synthetic"

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CalibrationArtifact:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema", 1) != ARTIFACT_SCHEMA:
            raise ValueError(
                f"Calibration artifact schema {data.get('schema')} != expected {ARTIFACT_SCHEMA}; "
                f"re-run `regime calibrate`."
            )
        return cls(**data)

    @classmethod
    def try_load(cls, path: Path) -> CalibrationArtifact | None:
        """Load if the file exists and parses; otherwise return None (detection stays raw)."""
        path = Path(path)
        if not path.exists():
            return None
        try:
            return cls.load(path)
        except Exception:  # noqa: BLE001 — a broken artifact must not break detection
            return None
