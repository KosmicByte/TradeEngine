"""
stochax_market/diagnostics.py

Post-hoc analysis of V3 (stochax-market) simulation and forecast results.
Reads the merged sim CSV produced by `stochax-merge` and the fitted
`params.pkl` produced by `stochax-fit`, runs a battery of statistical and
model-health checks, and emits a structured Markdown report describing
what's working, what isn't, and what to investigate.

The checks are organised into seven sections:

  1. Forecast Accuracy        — MAPE, RMSE, MAE, bias, Theil's U
  2. Directional Performance  — sign-match rate, hit rate by step
  3. Variance Diagnostics     — catches "predictions too flat" regressions
                                (drift / volatility bug signatures)
  4. Residual Analysis        — bias, autocorrelation, normality
  5. GARCH Health             — α+β persistence, unconditional variance,
                                fit-vs-realised correlation
  6. Drift Calibration        — annualised drift sanity check
  7. Calibration Convergence  — final loss, BFGS step-cap detection

Each finding carries a status (`ok` / `warn` / `fail`) and a one-line
diagnosis explaining what it means and (when applicable) what to try.

Usage
-----
Library:

    from stochax_market.diagnostics import analyze
    report = analyze(
        symbol      = "RELIANCE",
        sim_csv     = Path("RELIANCE_sim.csv"),
        params_path = Path("params.pkl"),
        out_md      = Path("RELIANCE_diagnostics.md"),
    )
    print(report.to_markdown())

CLI (after wiring into cli.py:diagnose_app):

    stochax-diagnose --symbol RELIANCE \\
                     --sim-csv RELIANCE_sim.csv \\
                     --params params.pkl \\
                     --out RELIANCE_diagnostics.md
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from stochax_market.data.loader import load_stock


# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════

_STATUS_GLYPH = {"ok": "✓", "warn": "⚠", "fail": "✗"}


@dataclass
class Finding:
    """
    One diagnostic check result.

    Attributes
    ----------
    metric    : Short metric name shown in the report (e.g. 'MAPE').
    value     : The measured value, formatted as a string for display.
    status    : 'ok' | 'warn' | 'fail'.
    diagnosis : One-line explanation of what the value means and, if not ok,
                what the likely cause is or how to investigate. Empty for
                ok-status rows.
    """
    metric:    str
    value:     str
    status:    str
    diagnosis: str = ""

    def to_row(self) -> str:
        glyph = _STATUS_GLYPH.get(self.status, "?")
        diag  = f" — {self.diagnosis}" if self.diagnosis else ""
        return f"| {glyph} | **{self.metric}** | `{self.value}` |{diag} |"


@dataclass
class DiagnosticsReport:
    """
    Structured output of `analyze()`.

    Findings are grouped into sections; rendering walks them in order.
    `summary` carries the headline numbers used by the executive summary.
    """
    symbol:   str
    n_total:  int
    n_realised: int
    sections: dict[str, list[Finding]] = field(default_factory=dict)
    summary:  dict[str, str] = field(default_factory=dict)

    def overall_status(self) -> str:
        """Worst status across all findings."""
        statuses = {f.status for s in self.sections.values() for f in s}
        if "fail" in statuses: return "fail"
        if "warn" in statuses: return "warn"
        return "ok"

    def recommendations(self) -> list[str]:
        """Auto-generated recommendations from any non-ok findings."""
        recs = []
        for section, findings in self.sections.items():
            for f in findings:
                if f.status in {"warn", "fail"} and f.diagnosis:
                    recs.append(f"**[{section}]** {f.metric}: {f.diagnosis}")
        return recs

    def to_markdown(self) -> str:
        glyph     = _STATUS_GLYPH[self.overall_status()]
        verdict   = {"ok": "Healthy", "warn": "Issues detected", "fail": "Significant problems"}[self.overall_status()]
        n_pending = self.n_total - self.n_realised

        lines = [
            f"# {self.symbol} — V3 Diagnostics Report",
            "",
            f"**Overall:** {glyph} {verdict}",
            "",
            f"- Forecast steps: {self.n_total}  ·  realised: {self.n_realised}  ·  forecast-only: {n_pending}",
        ]
        for k, v in self.summary.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

        for section, findings in self.sections.items():
            if not findings:
                continue
            lines.append(f"## {section}")
            lines.append("")
            lines.append("| Status | Metric | Value | Diagnosis |")
            lines.append("|:---:|:---|:---|:---|")
            for f in findings:
                lines.append(f.to_row())
            lines.append("")

        recs = self.recommendations()
        if recs:
            lines.append("## Recommendations")
            lines.append("")
            for r in recs:
                lines.append(f"- {r}")
            lines.append("")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(x: float, prec: int = 4) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    if abs(x) >= 1000:
        return f"{x:,.2f}"
    return f"{x:.{prec}f}"


def _pct(x: float) -> str:
    if x is None or np.isnan(x):
        return "n/a"
    return f"{x*100:.2f}%"


def _load_params(params_path: Path) -> dict:
    """Normalise pickled params into a canonical dict (matches visualize.py)."""
    with open(params_path, "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, tuple) and len(raw) == 3:
        spde, garch, L = raw
        return {"spde": spde, "garch": garch, "L": float(L)}
    raise TypeError(f"params.pkl has unrecognised structure: {type(raw)}")


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Forecast Accuracy
# ══════════════════════════════════════════════════════════════════════════════

def _accuracy_findings(realised: pd.DataFrame) -> list[Finding]:
    """
    Standard regression-style error metrics on the realised window.

    realised : DataFrame with columns predicted_price, actual_price, date.
    """
    out: list[Finding] = []
    if len(realised) < 2:
        out.append(Finding("MAPE", "n/a", "warn",
                           "fewer than 2 realised points; need more data to assess accuracy"))
        return out

    pred = realised["predicted_price"].to_numpy(dtype=float)
    act  = realised["actual_price"].to_numpy(dtype=float)
    err  = pred - act

    mape = float(np.mean(np.abs(err / act)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    # Status thresholds for daily equity prediction are intentionally lenient —
    # MAPE > 5% is poor for a stationary series at daily frequency.
    out.append(Finding(
        "MAPE", _pct(mape),
        "ok"   if mape < 0.02 else
        "warn" if mape < 0.05 else "fail",
        "" if mape < 0.02 else
        "5–10% is borderline; >10% suggests scale or drift miscalibration",
    ))
    out.append(Finding("RMSE (INR)", _fmt(rmse, 2), "ok", ""))
    out.append(Finding("MAE  (INR)", _fmt(mae, 2),  "ok", ""))

    # Bias as fraction of mean price; >1% systematic bias is meaningful.
    bias_frac = bias / float(np.mean(act))
    out.append(Finding(
        "Bias (mean error, INR)", _fmt(bias, 2),
        "ok"   if abs(bias_frac) < 0.01 else
        "warn" if abs(bias_frac) < 0.03 else "fail",
        "" if abs(bias_frac) < 0.01 else
        f"systematic {'over' if bias > 0 else 'under'}-prediction "
        f"of {abs(bias_frac)*100:.1f}% — check μ_scale calibration and drift handling",
    ))

    # Theil's U vs random-walk naive forecast: predict(t) = actual(t-1).
    # U < 1 means the model beats the naive baseline.
    if len(realised) >= 2:
        naive_err   = act[1:] - act[:-1]
        model_err   = err[1:]
        mse_naive   = float(np.mean(naive_err ** 2))
        mse_model   = float(np.mean(model_err ** 2))
        if mse_naive > 0:
            theil_u = float(np.sqrt(mse_model / mse_naive))
            out.append(Finding(
                "Theil's U vs random walk", _fmt(theil_u, 3),
                "ok"   if theil_u < 0.95 else
                "warn" if theil_u < 1.10 else "fail",
                "" if theil_u < 0.95 else
                f"{'roughly tied with' if theil_u < 1.10 else 'WORSE than'} naive RW baseline; "
                "the SPDE is not adding signal vs predict-yesterday's-price",
            ))

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Directional Performance
# ══════════════════════════════════════════════════════════════════════════════

def _directional_findings(realised: pd.DataFrame) -> list[Finding]:
    out: list[Finding] = []
    if len(realised) < 3:
        return out

    pred = realised["predicted_price"].to_numpy(dtype=float)
    act  = realised["actual_price"].to_numpy(dtype=float)

    pred_dir = np.sign(np.diff(pred))
    act_dir  = np.sign(np.diff(act))
    matches  = (pred_dir == act_dir) & (act_dir != 0)
    valid    = act_dir != 0
    rate     = float(matches.sum() / max(valid.sum(), 1))

    # Random guessing is 50%; below 45% indicates anti-correlation.
    out.append(Finding(
        "Directional accuracy (consecutive)", _pct(rate),
        "ok"   if rate >= 0.55 else
        "warn" if rate >= 0.45 else "fail",
        "" if rate >= 0.55 else
        "≤50% means the model can't tell up-days from down-days — "
        "examine drift sign and noise dominance",
    ))

    # Pearson correlation between consecutive returns.
    pred_ret = np.diff(pred) / pred[:-1]
    act_ret  = np.diff(act)  / act[:-1]
    if np.std(pred_ret) > 0 and np.std(act_ret) > 0:
        rho = float(np.corrcoef(pred_ret, act_ret)[0, 1])
        out.append(Finding(
            "Return correlation (ρ)", _fmt(rho, 3),
            "ok"   if rho >  0.20 else
            "warn" if rho > -0.05 else "fail",
            "" if rho > 0.20 else
            "near-zero or negative correlation between predicted and actual returns",
        ))

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — Variance Diagnostics
#
# This is where the constant-drift / frozen-volatility bug regressions show
# up: predictions become too flat relative to actual price movement, so the
# ratio of predicted std to actual std collapses toward zero.
# ══════════════════════════════════════════════════════════════════════════════

def _variance_findings(
    sim: pd.DataFrame,
    realised: pd.DataFrame,
) -> list[Finding]:
    out: list[Finding] = []
    if len(realised) < 3:
        return out

    pred_std = float(np.std(realised["predicted_price"].to_numpy(dtype=float)))
    act_std  = float(np.std(realised["actual_price"].to_numpy(dtype=float)))

    if act_std > 0:
        ratio = pred_std / act_std
        out.append(Finding(
            "Predicted/actual std ratio", _fmt(ratio, 3),
            "ok"   if 0.5 <= ratio <= 2.0 else
            "warn" if 0.3 <= ratio <= 3.0 else "fail",
            "" if 0.5 <= ratio <= 2.0 else
            (
                "predicted variance much smaller than actual — "
                "classic signature of frozen-volatility or constant-drift bugs in predict.py"
                if ratio < 0.5 else
                "predicted variance much larger than actual — "
                "GARCH unconditional vol may be inflated; check ω initialisation"
            ),
        ))

    # Range over the FULL forecast horizon (including forecast-only steps).
    # If the entire forecast is near-flat, that's a strong signal too.
    full_range = float(sim["predicted_price"].max() - sim["predicted_price"].min())
    full_mean  = float(sim["predicted_price"].mean())
    rel_range  = full_range / full_mean if full_mean > 0 else 0.0
    out.append(Finding(
        "Predicted relative range (full horizon)", _pct(rel_range),
        "ok"   if rel_range > 0.02 else
        "warn" if rel_range > 0.005 else "fail",
        "" if rel_range > 0.02 else
        "forecast is essentially a flat line over the full horizon — "
        "noise term may be too small or being dominated by deterministic drift",
    ))

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Residual Analysis
# ══════════════════════════════════════════════════════════════════════════════

def _residual_findings(realised: pd.DataFrame) -> list[Finding]:
    out: list[Finding] = []
    if len(realised) < 5:
        return out

    pred = realised["predicted_price"].to_numpy(dtype=float)
    act  = realised["actual_price"].to_numpy(dtype=float)
    res  = pred - act

    # Lag-1 autocorrelation of residuals; |ρ| > 0.3 indicates leftover signal.
    if np.std(res) > 0:
        r1 = float(np.corrcoef(res[:-1], res[1:])[0, 1])
        out.append(Finding(
            "Residual lag-1 autocorrelation", _fmt(r1, 3),
            "ok"   if abs(r1) < 0.30 else
            "warn" if abs(r1) < 0.50 else "fail",
            "" if abs(r1) < 0.30 else
            "strong residual autocorrelation — model is leaving exploitable structure on the table; "
            "consider a longer drift window or explicit AR component",
        ))

    # Skewness and excess kurtosis as cheap normality proxies.
    z   = (res - np.mean(res)) / (np.std(res) + 1e-12)
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4) - 3.0)
    out.append(Finding(
        "Residual skewness", _fmt(skew, 3),
        "ok" if abs(skew) < 1.0 else "warn",
        "" if abs(skew) < 1.0 else "asymmetric error distribution — possible regime-shift issue",
    ))
    out.append(Finding(
        "Residual excess kurtosis", _fmt(kurt, 3),
        "ok" if abs(kurt) < 3.0 else "warn",
        "" if abs(kurt) < 3.0 else "fat-tailed residuals — model under-estimates extreme moves",
    ))

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Section 5 — GARCH Health
# ══════════════════════════════════════════════════════════════════════════════

def _garch_findings(garch, df: pd.DataFrame) -> list[Finding]:
    """
    GARCH(1,1) model-level diagnostics. Requires the fitted module so we can
    call it on log-returns and pull α, β, ω.
    """
    import jax.numpy as jnp

    out: list[Finding] = []

    omega = float(garch.omega)
    alpha = float(garch.alpha)
    beta  = float(garch.beta)
    persistence = alpha + beta
    half_life   = float("inf") if persistence >= 1.0 else float(-np.log(2) / np.log(persistence))

    # α + β should be < 1 for stationarity; close to 1 means very persistent
    # shocks (typical for equities) but >1 is non-stationary and broken.
    out.append(Finding(
        "GARCH α + β (persistence)", _fmt(persistence, 4),
        "ok"   if persistence < 0.99 else
        "warn" if persistence < 1.00 else "fail",
        "" if persistence < 0.99 else
        ("α+β ≥ 1 means GARCH is non-stationary — refit with bounded β"
         if persistence >= 1.0 else
         "near-unit-root persistence; volatility shocks decay extremely slowly"),
    ))
    out.append(Finding(
        "Volatility half-life (days)",
        f"{half_life:.1f}" if half_life != float("inf") else "∞",
        "ok", ""
    ))
    out.append(Finding("GARCH ω", _fmt(omega, 6), "ok", ""))
    out.append(Finding("GARCH α", _fmt(alpha, 4), "ok", ""))
    out.append(Finding("GARCH β", _fmt(beta,  4), "ok", ""))

    # Unconditional variance ω/(1-α-β) compared to realised log-return variance.
    if persistence < 1.0:
        uncond_var = omega / (1.0 - persistence)
        log_ret    = np.log(df["Close"] / df["Close"].shift(1)).dropna().to_numpy()
        realised_var = float(np.var(log_ret))
        if realised_var > 0:
            ratio = uncond_var / realised_var
            out.append(Finding(
                "Unconditional / realised variance", _fmt(ratio, 3),
                "ok"   if 0.7 <= ratio <= 1.5 else
                "warn" if 0.4 <= ratio <= 2.5 else "fail",
                "" if 0.7 <= ratio <= 1.5 else
                f"GARCH long-run variance is {'inflated' if ratio > 1.5 else 'deflated'} "
                "relative to realised data — check ω initialisation (~−11.0 raw was the right anchor)",
            ))

        # Conditional vs realised correlation: how well does GARCH track
        # rolling realised vol?
        sigma_daily = np.array(garch(jnp.array(log_ret, dtype=jnp.float32)))
        rolling_vol = pd.Series(log_ret).rolling(30).std().to_numpy()
        mask = ~np.isnan(rolling_vol)
        if mask.sum() > 30 and np.std(sigma_daily[mask]) > 0 and np.std(rolling_vol[mask]) > 0:
            rho = float(np.corrcoef(sigma_daily[mask], rolling_vol[mask])[0, 1])
            out.append(Finding(
                "GARCH vs 30d realised vol (ρ)", _fmt(rho, 3),
                "ok"   if rho >  0.50 else
                "warn" if rho >  0.20 else "fail",
                "" if rho > 0.50 else
                "fitted GARCH conditional vol does not track realised rolling vol — "
                "the volatility model is essentially decoupled from observed dynamics",
            ))

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Section 6 — Drift Calibration
# ══════════════════════════════════════════════════════════════════════════════

def _drift_findings(df: pd.DataFrame) -> list[Finding]:
    out: list[Finding] = []
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna().to_numpy()
    if len(log_ret) < 30:
        return out

    mean_drift_daily  = float(np.mean(log_ret))
    mean_drift_annual = mean_drift_daily * 252

    # NIFTY equities historically show ~10–15% annualised drift; >40% is
    # unrealistic and probably indicates a window-of-fit anomaly.
    out.append(Finding(
        "Mean historical drift (annualised)", _pct(mean_drift_annual),
        "ok"   if abs(mean_drift_annual) < 0.40 else
        "warn" if abs(mean_drift_annual) < 0.80 else "fail",
        "" if abs(mean_drift_annual) < 0.40 else
        "annualised drift outside ±40% — fit window may be dominated by a single regime; "
        "consider a longer history or a rolling drift window",
    ))

    # Recent vs full-history drift divergence: a large gap means the recent
    # market regime differs substantially from training.
    recent = log_ret[-60:] if len(log_ret) >= 60 else log_ret
    recent_drift_annual = float(np.mean(recent)) * 252
    gap = abs(recent_drift_annual - mean_drift_annual)
    out.append(Finding(
        "Recent-60d drift (annualised)", _pct(recent_drift_annual),
        "ok"   if gap < 0.30 else
        "warn" if gap < 0.60 else "fail",
        "" if gap < 0.30 else
        "recent drift diverges substantially from full-history mean — "
        "regime shift; predictions anchored to historical mean may lag current trend",
    ))

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Section 7 — Calibration Convergence
# ══════════════════════════════════════════════════════════════════════════════

def _calibration_findings(saved: dict) -> list[Finding]:
    out: list[Finding] = []
    info = saved.get("loss_info", {}) or {}

    final_loss = info.get("final_loss")
    if final_loss is not None:
        fl = float(final_loss)
        out.append(Finding(
            "Final calibration loss", _fmt(fl, 6),
            "ok"   if fl < 0.01 else
            "warn" if fl < 0.10 else "fail",
            "" if fl < 0.01 else
            "loss above 0.01 suggests under-calibration; consider more BFGS steps or richer init",
        ))

    n_steps = info.get("n_steps") or info.get("steps_taken")
    if n_steps is not None:
        cap = 1000  # stochax-fit's documented default cap
        out.append(Finding(
            "BFGS steps taken", str(n_steps),
            "ok"   if int(n_steps) < cap else "warn",
            "" if int(n_steps) < cap else
            f"hit the {cap}-step cap — optimiser may not have fully converged; "
            "rerun with --n-steps higher or check loss landscape for plateaus",
        ))

    L = saved.get("L")
    if L is not None:
        out.append(Finding("Price scale L", _fmt(float(L), 2), "ok", ""))

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def analyze(
    symbol:      str,
    sim_csv:     Path,
    params_path: Optional[Path] = None,
    out_md:      Optional[Path] = None,
) -> DiagnosticsReport:
    """
    Run all diagnostic checks and produce a `DiagnosticsReport`.

    Parameters
    ----------
    symbol      : Stock ticker, used to load V1.1.0 historical data.
    sim_csv     : Merged sim CSV (cols: step, predicted_price, actual_price, date).
                  The output of `stochax-merge`.
    params_path : Optional path to fitted params.pkl. If provided, GARCH and
                  calibration-health sections are populated; otherwise they
                  are skipped.
    out_md      : Optional output path for the Markdown report. If None, the
                  report is returned but not written.

    Returns
    -------
    DiagnosticsReport with sections and headline summary populated.
    """
    sim_csv = Path(sim_csv)
    sim     = pd.read_csv(sim_csv)

    required = {"step", "predicted_price", "actual_price", "date"}
    missing  = required - set(sim.columns)
    if missing:
        raise ValueError(
            f"analyze: {sim_csv} is missing columns {missing}. "
            "Run `stochax-merge` first to produce the merged 4-column schema."
        )

    realised = sim.dropna(subset=["actual_price"]).reset_index(drop=True)

    df = load_stock(symbol)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(f"load_stock({symbol!r}) missing Date/Close columns.")
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    sections: dict[str, list[Finding]] = {}
    sections["Forecast Accuracy"]       = _accuracy_findings(realised)
    sections["Directional Performance"] = _directional_findings(realised)
    sections["Variance Diagnostics"]    = _variance_findings(sim, realised)
    sections["Residual Analysis"]       = _residual_findings(realised)
    sections["Drift Calibration"]       = _drift_findings(df)

    if params_path is not None and Path(params_path).exists():
        saved = _load_params(Path(params_path))
        if "garch" in saved:
            sections["GARCH Health"] = _garch_findings(saved["garch"], df)
        sections["Calibration Convergence"] = _calibration_findings(saved)

    summary: dict[str, str] = {}
    if len(realised) >= 2:
        pred = realised["predicted_price"].to_numpy(dtype=float)
        act  = realised["actual_price"].to_numpy(dtype=float)
        mape = float(np.mean(np.abs((pred - act) / act)))
        summary["MAPE (realised)"] = _pct(mape)
        summary["Mean predicted"]  = f"₹{_fmt(float(pred.mean()), 2)}"
        summary["Mean actual"]     = f"₹{_fmt(float(act.mean()), 2)}"

    report = DiagnosticsReport(
        symbol     = symbol,
        n_total    = len(sim),
        n_realised = len(realised),
        sections   = sections,
        summary    = summary,
    )

    if out_md is not None:
        out_md = Path(out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(report.to_markdown(), encoding="utf-8")

    return report
