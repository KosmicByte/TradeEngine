"""Transition-risk scoring: how likely is the current regime to change soon?

Builds on three signals — measured against a rolling baseline so we react to *changes*
in the structure of the dynamics, not just to absolute values:

    1. Eigenvalue drift     — L2 distance between the current top-k eigenvalues
                              (sorted by magnitude) and the average of recent windows.
    2. Spectral gap collapse — drop in |λ₁| - |λ₂| vs. baseline. A collapsing gap means
                               the dominant mode is losing its dominance.
    3. Vol acceleration      — z-score of the recent change in realized vol.

We combine them into one [0, 1] score with a logistic squash.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from stochax_market.regime.core.edmd import fit_edmd
from stochax_market.regime.core.observables import log_returns, rolling_std
from stochax_market.regime.models import TransitionRisk


def _sigmoid(x: float, scale: float = 1.0) -> float:
    return float(1.0 / (1.0 + np.exp(-x * scale)))


def _eigenvalue_drift(current: np.ndarray, baseline: list[np.ndarray]) -> float:
    """L2 distance between current top eigenvalues and the mean of baseline windows."""
    if not baseline:
        return 0.0
    k = min(len(current), min(len(b) for b in baseline))
    if k == 0:
        return 0.0
    cur = current[:k]
    base_mean = np.mean(np.stack([b[:k] for b in baseline]), axis=0)
    return float(np.linalg.norm(cur - base_mean))


def score_transition_risk(
    close: np.ndarray,
    timestamps: pd.Series | None = None,
    symbol: str = "?",
    interval: str = "1d",
    window: int = 126,
    baseline_windows: int = 5,
    edmd_rank: int = 10,
    threshold: float = 0.6,
) -> TransitionRisk:
    """Estimate transition risk.

    Args:
        close: 1-D price series, length T.
        window: bars per EDMD fit.
        baseline_windows: number of historical windows used to form the baseline.
        edmd_rank: rank for EDMD.
        threshold: risk above this is reported as `crossed_threshold=True`.
    """
    close = np.asarray(close, dtype=float)
    if len(close) < window * (baseline_windows + 1):
        # Soft-fail: just compute what we can.
        baseline_windows = max(1, (len(close) // window) - 1)

    # Current spectrum
    cur_edmd = fit_edmd(close[-window:], rank=edmd_rank)
    cur_eigs = cur_edmd.eigenvalues
    cur_gap = cur_edmd.spectral_gap

    # Baseline spectra: equally-spaced windows preceding the current one.
    baseline_eigs: list[np.ndarray] = []
    baseline_gaps: list[float] = []
    for i in range(1, baseline_windows + 1):
        end = len(close) - i * (window // 2)
        start = end - window
        if start < 50:
            break
        try:
            b_edmd = fit_edmd(close[start:end], rank=edmd_rank)
            baseline_eigs.append(b_edmd.eigenvalues)
            baseline_gaps.append(b_edmd.spectral_gap)
        except (ValueError, np.linalg.LinAlgError):
            continue

    eig_drift = _eigenvalue_drift(cur_eigs, baseline_eigs)
    baseline_gap_mean = float(np.mean(baseline_gaps)) if baseline_gaps else cur_gap
    gap_collapse = max(0.0, baseline_gap_mean - cur_gap)

    # Vol acceleration
    r = log_returns(close)
    vol_short = rolling_std(r, 21)
    vol_long = rolling_std(r, 63)
    vs = vol_short[~np.isnan(vol_short)]
    vl = vol_long[~np.isnan(vol_long)]
    if len(vs) >= 20 and len(vl) >= 20:
        recent_diff = float(vs[-1] - vl[-1])
        hist_diff = vs[-60:] - vl[-60:]
        mu = float(np.mean(hist_diff))
        sigma = float(np.std(hist_diff)) or 1e-8
        vol_accel = (recent_diff - mu) / sigma
    else:
        vol_accel = 0.0

    # Combine. Each component is mapped to ~[0, 1] then averaged.
    s_eig = _sigmoid(eig_drift - 0.5, scale=3.0)
    s_gap = _sigmoid(gap_collapse * 10 - 1.0, scale=2.0)
    s_vol = _sigmoid(abs(vol_accel) - 1.0, scale=1.5)
    risk = float(np.clip(0.4 * s_eig + 0.3 * s_gap + 0.3 * s_vol, 0.0, 1.0))

    # Narrative
    pieces = []
    if s_eig > 0.5:
        pieces.append(f"eigenvalue drift {eig_drift:.2f}")
    if s_gap > 0.5:
        pieces.append(f"spectral gap collapsed by {gap_collapse:.3f}")
    if s_vol > 0.5:
        pieces.append(f"vol z-score {vol_accel:+.2f}")
    note = (
        ("Elevated transition risk — " + ", ".join(pieces) + ".") if pieces else "Regime appears stable."
    )

    as_of = (
        pd.to_datetime(timestamps.iloc[-1]).to_pydatetime()
        if timestamps is not None and len(timestamps) > 0
        else datetime.now()
    )

    return TransitionRisk(
        symbol=symbol,
        interval=interval,
        as_of=as_of,
        risk_score=risk,
        crossed_threshold=risk >= threshold,
        eigenvalue_drift=eig_drift,
        spectral_gap_collapse=gap_collapse,
        vol_acceleration=vol_accel,
        note=note,
    )
