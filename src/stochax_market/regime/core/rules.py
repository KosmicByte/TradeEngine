"""Simple, transparent rule-based regime classifier.

Acts as the third independent vote in our ensemble. Uses only annualised drift and
realized vol with hard thresholds — no learning, fully explainable, fast.

Why this matters: ensembling EDMD (linear operator), HMM (probabilistic states), and a
rule classifier (statistical thresholds) gives us *uncorrelated* errors, which is what
ensembles need to beat their best constituent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stochax_market.regime.core.observables import log_returns, rolling_std
from stochax_market.regime.models import RegimeLabel

# Annualisation: 252 trading days, NSE convention.
TRADING_DAYS = 252


def variance_ratio(returns: np.ndarray, k: int = 5) -> float:
    """Lo-MacKinlay variance ratio test.

    VR(k) = Var(P_{t+k} - P_t) / (k * Var(P_{t+1} - P_t))
      = 1   → random walk
      > 1   → trending / momentum (variance grows faster than linearly with k)
      < 1   → mean-reverting (variance grows slower than linearly with k)

    On log returns, P_{t+k} - P_t = sum of k consecutive returns, so we compute
    Var of k-step rolling sums divided by k * Var of single-step returns.

    Args:
        returns: 1-D array of single-period log returns.
        k: aggregation horizon. Larger k = more sensitive to long-horizon reversion
           but noisier estimate. k=5 is a standard short-horizon choice.

    Returns:
        VR statistic. Values < 0.8 are reasonable evidence of mean reversion;
        values > 1.2 of momentum. ≈ 1 means indistinguishable from random walk.
    """
    if len(returns) < k * 3:
        return 1.0
    var1 = float(np.var(returns, ddof=1))
    if var1 < 1e-12:
        return 1.0
    # k-period cumulative returns (overlapping)
    csum = np.cumsum(np.insert(returns, 0, 0.0))
    k_step = csum[k:] - csum[:-k]
    var_k = float(np.var(k_step, ddof=1))
    return var_k / (k * var1)


def ar1_half_life(close: np.ndarray) -> tuple[float, float]:
    """AR(1) fit on log-price level → half-life of mean reversion.

    For an OU process dx = θ(μ - x)dt + σ dW, the discrete AR(1) representation is
        x_{t+1} = (1 - θΔt) x_t + θΔt μ + ε
    so the AR(1) slope φ = (1 - θΔt) directly encodes reversion speed:
      φ ≈ 1.00 → random walk (no reversion at price level)
      φ < 0.95 → meaningful mean reversion
      φ < 0.90 → strong mean reversion

    Half-life (bars) = ln(2) / -ln(φ) is the expected time for the deviation from
    mean to shrink by 50%.

    This is the proper price-level reversion test — returns-based tests like VR
    cannot see OU-style reversion because it is encoded in the price level, not
    in return autocorrelation.

    Returns:
        (phi, half_life_in_bars). half_life = inf when phi >= 1.
    """
    if len(close) < 30:
        return 1.0, float("inf")
    x = np.log(np.asarray(close, dtype=float))
    x_lag = x[:-1]
    x_now = x[1:]
    # Centred regression handles drift; slope is the AR(1) coefficient.
    x_lag_c = x_lag - np.mean(x_lag)
    x_now_c = x_now - np.mean(x_now)
    denom = float(np.sum(x_lag_c * x_lag_c))
    if denom < 1e-12:
        return 1.0, float("inf")
    phi = float(np.sum(x_lag_c * x_now_c) / denom)
    if phi >= 0.9999:
        return phi, float("inf")
    if phi <= 0.0:
        return phi, 1.0
    half_life = float(np.log(2.0) / -np.log(phi))
    return phi, half_life


@dataclass(frozen=True)
class RuleResult:
    """Output of the rule-based classifier."""

    label: RegimeLabel
    confidence: float
    annualised_drift: float  # μ × 252
    annualised_vol: float  # σ × sqrt(252)
    trend_strength: float  # μ / σ (Sharpe-like, daily)
    variance_ratio_5: float  # VR(5) — return-space mean-reversion / momentum diagnostic
    ar1_phi: float  # AR(1) coefficient on log-price
    half_life_days: float  # ln(2)/-ln(phi) — price-level reversion half-life
    note: str


def classify_rule_based(
    close: np.ndarray,
    window: int = 126,
    hist_vol_window: int = 252,
) -> RuleResult:
    """Classify the *current* regime from the last `window` bars.

    Args:
        close: 1-D price series.
        window: bars used to compute current drift & vol.
        hist_vol_window: longer window used as a vol baseline.

    Returns:
        RuleResult.
    """
    r = log_returns(close)
    # Adaptively shrink window if data is too short — never raise on insufficient data.
    if len(r) < window + 5:
        window = max(30, len(r) - 5)
    if len(r) < 30:
        raise ValueError(f"Need at least 30 returns, got {len(r)}.")

    recent = r[-window:]
    mu = float(np.mean(recent))
    sigma = float(np.std(recent))
    ann_drift = mu * TRADING_DAYS
    ann_vol = sigma * np.sqrt(TRADING_DAYS)
    trend = mu / sigma if sigma > 1e-12 else 0.0

    # Variance ratio at lag-5 — proper mean-reversion / momentum test on recent returns.
    # On a 126-bar window VR has noise σ ~ 0.10, so we require VR<0.70 (3σ below 1.0)
    # for "strong" mean-reversion evidence to avoid false positives on random walks.
    vr5 = variance_ratio(recent, k=5)
    is_mean_reverting_vr = vr5 < 0.70  # strong reversion signature (3σ on 126-bar windows)
    is_trending_vr = vr5 > 1.30  # strong momentum signature

    # AR(1) half-life on log-price — proper PRICE-LEVEL mean-reversion test.
    # Returns-based tests cannot see OU-style reversion (it lives in the price level).
    # Half-life < 15 days = strong reversion; 15-30 = mild; > 30 = essentially random walk.
    # We additionally require ann_vol >= 0.10 because in very-low-vol grinds the AR(1)
    # phi can appear low purely from noise, not from genuine reversion. True OU
    # reversion produces meaningful price deviations from mean, hence moderate vol.
    recent_close = close[-(window + 1) :]
    phi, half_life = ar1_half_life(recent_close)
    is_strong_price_reversion = half_life < 15.0 and ann_vol >= 0.10
    is_mild_price_reversion = 15.0 <= half_life < 30.0 and ann_vol >= 0.10

    # Vol percentile relative to a longer history.
    hist_vol = rolling_std(r, hist_vol_window)
    hist_valid = hist_vol[~np.isnan(hist_vol)]
    vol_pct = (
        float(np.mean(hist_valid < sigma)) if len(hist_valid) > 0 else 0.5
    )  # fraction of history with lower vol than now

    # Decision logic — order matters; first match wins.
    # ABSOLUTE thresholds dominate (catch self-contained high/low-vol series).
    # Within each vol regime we apply Sharpe thresholds calibrated to that vol level:
    # high-vol regimes need a STRONG Sharpe (>0.25) to declare a trend, because a
    # weak Sharpe at 50% ann vol is statistically indistinguishable from noise.
    is_high_vol_abs = ann_vol > 0.30  # > 30% annualised → unambiguously high
    is_low_vol_abs = ann_vol < 0.08  # < 8% annualised → unambiguously low

    # HIGHEST PRIORITY: strong AR(1) price-level reversion. This catches OU regimes
    # that returns-based tests cannot see. Requires (a) short half-life, (b) lack of
    # unambiguous trend signal (Sharpe < 0.25), and (c) modest annual drift (< 15%),
    # so we don't override genuine trends that happen to look noisy.
    if is_strong_price_reversion and abs(trend) < 0.25 and abs(ann_drift) < 0.15:
        label = RegimeLabel.MEAN_REVERTING
        note = (
            f"AR(1) φ={phi:.3f}, half-life={half_life:.1f} bars — strong price-level reversion."
        )
    # NEXT: strong variance-ratio evidence of mean reversion (return-space signal)
    elif is_mean_reverting_vr and abs(trend) < 0.20 and not is_high_vol_abs:
        label = RegimeLabel.MEAN_REVERTING
        note = f"Variance ratio VR(5)={vr5:.2f}<0.70 — strong return-space mean-reversion."
    elif is_high_vol_abs:
        if trend < -0.25:
            label = RegimeLabel.TRENDING_DOWN
            note = f"High vol (ann {ann_vol:.0%}) + strong negative Sharpe {trend:.2f}."
        elif trend > 0.25:
            label = RegimeLabel.BREAKOUT
            note = f"High vol (ann {ann_vol:.0%}) + strong positive Sharpe {trend:.2f}."
        else:
            label = RegimeLabel.HIGH_VOL_CHOP
            note = f"High vol (ann {ann_vol:.0%}), weak Sharpe {trend:+.2f} — directionless."
    elif is_low_vol_abs:
        if trend > 0.20:
            label = RegimeLabel.TRENDING_UP
            note = f"Low vol (ann {ann_vol:.0%}) + positive Sharpe {trend:.2f}."
        elif trend < -0.20:
            label = RegimeLabel.TRENDING_DOWN
            note = f"Low vol (ann {ann_vol:.0%}) + negative Sharpe {trend:.2f}."
        else:
            label = RegimeLabel.LOW_VOL_GRIND
            note = f"Low vol (ann {ann_vol:.0%}), weak Sharpe {trend:+.2f} — calm grind."
    # In MODERATE vol (8-30% ann), trend signal takes priority over vol percentile.
    # Require VR support for trends to avoid mistaking noise for momentum.
    elif trend > 0.10 and not is_mean_reverting_vr:
        label = RegimeLabel.TRENDING_UP
        note = f"Positive Sharpe {trend:.2f}, VR(5)={vr5:.2f}, ann drift {ann_drift:.1%}."
    elif trend < -0.10 and not is_mean_reverting_vr:
        label = RegimeLabel.TRENDING_DOWN
        note = f"Negative Sharpe {trend:.2f}, VR(5)={vr5:.2f}, ann drift {ann_drift:.1%}."
    # Only fall to percentile-based vol detection when no clear directional signal.
    elif vol_pct > 0.85 and trend > 0.05:
        label = RegimeLabel.BREAKOUT
        note = f"Elevated vol (top {(1-vol_pct)*100:.0f}%) + positive Sharpe {trend:.2f}."
    elif vol_pct > 0.85 and trend < -0.05:
        label = RegimeLabel.TRENDING_DOWN
        note = f"Elevated vol (top {(1-vol_pct)*100:.0f}%) + negative Sharpe {trend:.2f}."
    elif vol_pct > 0.80:
        label = RegimeLabel.HIGH_VOL_CHOP
        note = f"Elevated vol (top {(1-vol_pct)*100:.0f}%), no clear direction."
    elif vol_pct < 0.20 and abs(ann_drift) < 0.05:
        label = RegimeLabel.LOW_VOL_GRIND
        note = f"Suppressed vol (bottom {vol_pct*100:.0f}%), drift {ann_drift:.1%}."
    # Final fallthrough: weak Sharpe with neutral VR.
    # Use ABSOLUTE annual drift as tiebreaker — a 10%+ annual drift is meaningful
    # even when daily Sharpe is below threshold (small daily mean, but consistent).
    # Only call mean_reverting when VR really indicates it (VR < 0.85, halfway to threshold).
    elif vr5 < 0.85 and abs(ann_drift) < 0.10:
        label = RegimeLabel.MEAN_REVERTING
        note = f"VR(5)={vr5:.2f}<0.85 with weak drift {ann_drift:+.1%} — likely mean reversion."
    elif ann_drift > 0.08:
        label = RegimeLabel.TRENDING_UP
        note = f"Weak Sharpe {trend:+.2f} but meaningful annual drift {ann_drift:+.1%}."
    elif ann_drift < -0.08:
        label = RegimeLabel.TRENDING_DOWN
        note = f"Weak Sharpe {trend:+.2f} but meaningful annual drift {ann_drift:+.1%}."
    else:
        label = RegimeLabel.MEAN_REVERTING
        note = f"No clear signal (Sharpe={trend:+.2f}, VR={vr5:.2f}, drift={ann_drift:+.1%})."

    # Confidence: how far we are from the decision boundary.
    if label in (RegimeLabel.TRENDING_UP, RegimeLabel.TRENDING_DOWN):
        # VR or short half-life contradicts trend signal → cap confidence.
        base = min(1.0, abs(trend) / 0.3)
        if is_mean_reverting_vr or is_strong_price_reversion:
            base *= 0.5
        elif is_trending_vr:
            base = min(1.0, base * 1.3)
        confidence = float(base)
    elif label == RegimeLabel.HIGH_VOL_CHOP:
        # At very high abs vol, confidence is high regardless of percentile.
        if is_high_vol_abs:
            confidence = float(min(1.0, 0.6 + (ann_vol - 0.30) * 2))
        else:
            confidence = float(min(1.0, (vol_pct - 0.5) * 2))
    elif label == RegimeLabel.LOW_VOL_GRIND:
        if is_low_vol_abs:
            confidence = float(min(1.0, 0.7 + (0.08 - ann_vol) * 5))
        else:
            confidence = float(min(1.0, (0.5 - vol_pct) * 2))
    elif label == RegimeLabel.BREAKOUT:
        confidence = float(min(1.0, (vol_pct - 0.5) * 1.5 + abs(trend)))
    elif label == RegimeLabel.MEAN_REVERTING:
        # Short half-life is the strongest possible evidence: scale aggressively.
        if is_strong_price_reversion:
            # half_life in [1, 15] → confidence in [0.95, 0.65]
            confidence = float(min(1.0, max(0.65, 0.95 - (half_life - 1.0) * 0.02)))
        else:
            # Fall back to VR-based scaling
            confidence = float(min(1.0, max(0.5, (1.0 - vr5) * 3)))
    else:
        confidence = 0.5

    return RuleResult(
        label=label,
        confidence=confidence,
        annualised_drift=ann_drift,
        annualised_vol=ann_vol,
        trend_strength=trend,
        variance_ratio_5=vr5,
        ar1_phi=phi,
        half_life_days=half_life,
        note=note,
    )
