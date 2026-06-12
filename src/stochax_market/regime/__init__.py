"""Regime Radar — HMM + EDMD (Koopman) + rule-based ensemble market-state classification.

Ported from the standalone RegimeRadar repo into Project Ares Alpha. The pure-math core
(core/) has zero I/O coupling; calibration/, contract, version, and provenance provide the
R0/R1 reproducibility and uncertainty layers. Data binding is provided by stochax_market.data
at integration time (replacing the old yfinance/marketlake data.py).
"""
from stochax_market.regime.models import (
    RegimeLabel,
    RegimeResult,
    TransitionRisk,
    KoopmanModes,
    RegimeReason,
)
from stochax_market.regime.core.regime import detect_regime
from stochax_market.regime.core.transition import score_transition_risk

__all__ = [
    "RegimeLabel",
    "RegimeResult",
    "TransitionRisk",
    "KoopmanModes",
    "RegimeReason",
    "detect_regime",
    "score_transition_risk",
]
