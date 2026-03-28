"""
stochax_market/visualize.py

Diagnostic and results visualisation for the SPDE stock model.
All plots are saved as PNG. Call via CLI or import directly.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

try:
    pio.templates.default = "perplexity"
except Exception:
    pass

# ── Colour palette ────────────────────────────────────────────────────────────
_C = dict(
    close   = "#5B8DEF",
    vwap    = "#A78BFA",
    sim     = "#F97316",
    pred    = "#22C55E",
    vol     = "#F97316",
    band    = "rgba(34,197,94,0.15)",
    hlband  = "rgba(91,141,239,0.10)",
)

_LEGEND = dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5)


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path))


def _subtitle(text: str) -> str:
    return f"<br><span style='font-size:15px;font-weight:normal;'>{text}</span>"


# ══════════════════════════════════════════════════════════════════════════════
# Public plot functions
# ══════════════════════════════════════════════════════════════════════════════

def plot_historical(
    df: pd.DataFrame,
    symbol: str,
    out: Path,
) -> Path:
    """
    Full historical Close + VWAP time series.

    Parameters
    ----------
    df      : DataFrame with columns Date, Close, VWAP (full history).
    symbol  : Stock ticker label.
    out     : Output PNG path.

    Returns
    -------
    Path to saved PNG.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        mode="lines", name="Close",
        line=dict(color=_C["close"], width=1.2),
        fill="tozeroy", fillcolor="rgba(91,141,239,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["VWAP"],
        mode="lines", name="VWAP",
        line=dict(color=_C["vwap"], width=1, dash="dot"),
    ))

    start = df["Date"].dt.year.min()
    end   = df["Date"].dt.year.max()
    lo    = df["Close"].min()
    hi    = df["Close"].max()

    fig.update_layout(
        title=dict(text=(
            f"{symbol} Close & VWAP ({start}–{end})"
            + _subtitle(f"Source: NIFTY50 | ₹{lo:.0f} → ₹{hi:.0f} over {end-start} years")
        )),
        legend=_LEGEND,
    )
    fig.update_xaxes(title_text="Date", dtick="M24", tickformat="%Y")
    fig.update_yaxes(title_text="Price (INR)", tickformat=",.0f")

    _save(fig, out)
    return out


def plot_prediction(
    df: pd.DataFrame,
    symbol: str,
    pred_mean: list[float],
    pred_lo: list[float],
    pred_hi: list[float],
    pred_dates: pd.DatetimeIndex,
    out: Path,
    recent_n: int = 60,
) -> Path:
    """
    Recent N trading days of actual data overlaid with forward prediction
    and 90% confidence interval band.

    Parameters
    ----------
    df          : Full historical DataFrame.
    symbol      : Stock ticker label.
    pred_mean   : List of predicted mean prices, length = horizon.
    pred_lo     : List of 5th-percentile prices.
    pred_hi     : List of 95th-percentile prices.
    pred_dates  : Business-day DatetimeIndex of length horizon.
    out         : Output PNG path.
    recent_n    : Number of past trading days to show (default 60).

    Returns
    -------
    Path to saved PNG.
    """
    recent    = df[["Date", "Close", "High", "Low", "VWAP"]].tail(recent_n).copy()
    last_date = df["Date"].iloc[-1]
    horizon   = len(pred_mean)

    # Actual CI band (High-Low range)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(recent["Date"]) + list(recent["Date"])[::-1],
        y=list(recent["High"]) + list(recent["Low"])[::-1],
        fill="toself", fillcolor=_C["hlband"],
        line=dict(color="rgba(0,0,0,0)"), name="Daily H-L Range",
    ))
    fig.add_trace(go.Scatter(
        x=recent["Date"], y=recent["Close"],
        mode="lines", name="Actual Close",
        line=dict(color=_C["close"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=recent["Date"], y=recent["VWAP"],
        mode="lines", name="Actual VWAP",
        line=dict(color=_C["vwap"], width=1.5, dash="dot"),
    ))

    # Prediction CI band
    fig.add_trace(go.Scatter(
        x=list(pred_dates) + list(pred_dates[::-1]),
        y=list(pred_hi) + list(pred_lo[::-1]),
        fill="toself", fillcolor=_C["band"],
        line=dict(color="rgba(0,0,0,0)"), name="Pred 90% CI",
    ))
    fig.add_trace(go.Scatter(
        x=pred_dates, y=pred_mean,
        mode="lines+markers", name="Predicted Close",
        line=dict(color=_C["pred"], width=2.5),
        marker=dict(size=7),
    ))

    # Divider at last actual date
    fig.add_shape(
        type="line",
        x0=str(last_date), x1=str(last_date), y0=0, y1=1, yref="paper",
        line=dict(dash="dash", color="gray", width=1.5),
    )
    fig.add_annotation(
        x=str(last_date), y=0.97, yref="paper",
        text="Last Actual", showarrow=False,
        xanchor="right", font=dict(size=11, color="gray"),
    )

    fig.update_layout(
        title=dict(text=(
            f"{symbol}: Last {recent_n} Days + {horizon}-Day Prediction"
            + _subtitle(f"Source: NIFTY50 | Green band = 90% CI from 100 Monte Carlo paths")
        )),
        legend=_LEGEND,
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price (INR)", tickformat=",.0f")

    _save(fig, out)
    return out


def plot_sim_vs_actual(
    df: pd.DataFrame,
    symbol: str,
    sim_csv: Path,
    out: Path,
) -> Path:
    """
    Overlay SPDE simulation output against actual prices for the same
    number of timesteps, aligned from the most recent date backwards.

    Parameters
    ----------
    df      : Full historical DataFrame.
    symbol  : Stock ticker label.
    sim_csv : Path to simulation CSV (columns: step, predicted_price).
    out     : Output PNG path.

    Returns
    -------
    Path to saved PNG.
    """
    sim = pd.read_csv(sim_csv)
    n   = len(sim)

    # Align to most recent N actual days (not first N — avoids year-2000 bug)
    actual = df[["Date", "Close", "High", "Low"]].tail(n).reset_index(drop=True)

    fig = go.Figure()

    # Actual H-L band
    fig.add_trace(go.Scatter(
        x=list(actual["Date"]) + list(actual["Date"])[::-1],
        y=list(actual["High"]) + list(actual["Low"])[::-1],
        fill="toself", fillcolor=_C["hlband"],
        line=dict(color="rgba(0,0,0,0)"), name="Actual H-L Range",
    ))
    fig.add_trace(go.Scatter(
        x=actual["Date"], y=actual["Close"],
        mode="lines", name="Actual Close",
        line=dict(color=_C["close"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=actual["Date"], y=sim["predicted_price"],
        mode="lines", name="SPDE Simulation",
        line=dict(color=_C["sim"], width=2, dash="dash"),
    ))

    fig.update_layout(
        title=dict(text=(
            f"{symbol}: SPDE Simulation vs Actual ({n} steps)"
            + _subtitle("Orange dashed = model output | Blue = actual Close")
        )),
        legend=_LEGEND,
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price (INR)", tickformat=",.0f")

    _save(fig, out)
    return out


def plot_log_returns(
    df: pd.DataFrame,
    symbol: str,
    out: Path,
) -> Path:
    """
    Histogram of daily log-returns vs fitted normal distribution.
    Annotates excess kurtosis to motivate GARCH/SPDE modelling.

    Parameters
    ----------
    df      : DataFrame with Close column.
    symbol  : Stock ticker label.
    out     : Output PNG path.

    Returns
    -------
    Path to saved PNG.
    """
    lr   = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    xn   = np.linspace(float(lr.min()), float(lr.max()), 400)
    yn   = ((1 / (lr.std() * np.sqrt(2 * np.pi)))
            * np.exp(-0.5 * ((xn - lr.mean()) / lr.std()) ** 2))
    kurt = float(lr.kurtosis())

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=lr, nbinsx=150, histnorm="probability density",
        name="Log-Returns", marker_color=_C["close"], opacity=0.75,
    ))
    fig.add_trace(go.Scatter(
        x=xn, y=yn, mode="lines", name="Normal Fit",
        line=dict(color=_C["sim"], width=2.5, dash="dash"),
    ))

    fig.update_layout(
        title=dict(text=(
            f"{symbol} Log-Return Distribution (Excess Kurtosis = {kurt:.1f})"
            + _subtitle("Fat tails vs Normal → justifies GARCH volatility + SPDE noise")
        )),
        legend=_LEGEND,
    )
    fig.update_xaxes(title_text="Log Return")
    fig.update_yaxes(title_text="Density")

    _save(fig, out)
    return out


def plot_volatility(
    df: pd.DataFrame,
    symbol: str,
    out: Path,
    window: int = 30,
) -> Path:
    """
    Rolling annualised realised volatility, showing volatility clustering.

    Parameters
    ----------
    df      : DataFrame with Close column.
    symbol  : Stock ticker label.
    out     : Output PNG path.
    window  : Rolling window in trading days (default 30).

    Returns
    -------
    Path to saved PNG.
    """
    df = df.copy()
    df["lr"]   = np.log(df["Close"] / df["Close"].shift(1))
    df["rvol"] = df["lr"].rolling(window).std() * np.sqrt(252)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["rvol"],
        mode="lines", name=f"{window}d Vol",
        line=dict(color=_C["vol"], width=1.5),
        fill="tozeroy", fillcolor="rgba(249,115,22,0.10)",
    ))

    start = df["Date"].dt.year.min()
    end   = df["Date"].dt.year.max()

    fig.update_layout(
        title=dict(text=(
            f"{symbol} Volatility Clustering ({start}–{end})"
            + _subtitle(f"{window}-day rolling annualised vol | Spikes = market stress events")
        )),
        legend=_LEGEND,
    )
    fig.update_xaxes(title_text="Date", dtick="M24", tickformat="%Y")
    fig.update_yaxes(title_text="Ann. Volatility")

    _save(fig, out)
    return out


def plot_garch_vol_fit(
    df: pd.DataFrame,
    symbol: str,
    params_path: Path,
    out: Path,
) -> Path:
    """
    Compare GARCH(1,1) fitted conditional volatility against realised
    rolling volatility. Requires calibrated params.pkl.

    Parameters
    ----------
    df           : DataFrame with Close column.
    symbol       : Stock ticker label.
    params_path  : Path to params.pkl produced by stochax-fit.
    out          : Output PNG path.

    Returns
    -------
    Path to saved PNG.
    """
    import jax.numpy as jnp
    from stochax_market.model.volatility import GARCHVolatility

    with open(params_path, "rb") as f:
        saved = pickle.load(f)

    garch: GARCHVolatility = saved["garch"]
    lr    = np.log(df["Close"] / df["Close"].shift(1)).dropna().values
    dates = df["Date"].iloc[-len(lr):]

    sigma2 = np.array(garch(jnp.array(lr, dtype=jnp.float32)))
    sigma_fitted  = np.sqrt(sigma2) * np.sqrt(252)

    rolling_vol = (pd.Series(lr).rolling(30).std() * np.sqrt(252)).values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=rolling_vol,
        mode="lines", name="30d Realised Vol",
        line=dict(color=_C["close"], width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=sigma_fitted,
        mode="lines", name="GARCH(1,1) Fitted Vol",
        line=dict(color=_C["sim"], width=1.5),
        fill="tozeroy", fillcolor="rgba(249,115,22,0.07)",
    ))

    fig.update_layout(
        title=dict(text=(
            f"{symbol}: GARCH(1,1) vs Realised Volatility"
            + _subtitle("Orange = GARCH fitted | Blue dashed = 30d rolling realised")
        )),
        legend=_LEGEND,
    )
    fig.update_xaxes(title_text="Date", dtick="M24", tickformat="%Y")
    fig.update_yaxes(title_text="Ann. Volatility")

    _save(fig, out)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: run all plots in one call
# ══════════════════════════════════════════════════════════════════════════════

def run_all(
    symbol: str,
    sim_csv: Optional[Path] = None,
    params_path: Optional[Path] = None,
    horizon: int = 21,
    recent_n: int = 60,
    out_dir: Path = Path("plots"),
    seed: int = 42,
) -> dict[str, Path]:
    """
    Generate all diagnostic plots for a given symbol.
    Calls stochax_market.data.loader and stochax_market.predict internally.

    Parameters
    ----------
    symbol      : NIFTY50 stock symbol (e.g. 'RELIANCE').
    sim_csv     : Path to simulation CSV; skips sim plot if None.
    params_path : Path to params.pkl; skips GARCH + prediction plots if None.
    horizon     : Forecast horizon in trading days (default 21).
    recent_n    : Past days shown in prediction chart (default 60).
    out_dir     : Directory to write all PNGs.
    seed        : Random seed for MC prediction.

    Returns
    -------
    Dict mapping plot name → Path of saved PNG.
    """
    from stochax_market.data.loader import load_stock

    df      = load_stock(symbol)
    out_dir = Path(out_dir)
    saved   = {}

    # 1. Historical
    saved["historical"] = plot_historical(
        df, symbol, out_dir / f"{symbol}_historical.png"
    )

    # 2. Log-returns
    saved["log_returns"] = plot_log_returns(
        df, symbol, out_dir / f"{symbol}_log_returns.png"
    )

    # 3. Volatility clustering
    saved["volatility"] = plot_volatility(
        df, symbol, out_dir / f"{symbol}_volatility.png"
    )

    # 4. Simulation vs actual (if sim CSV provided)
    if sim_csv is not None and Path(sim_csv).exists():
        saved["sim_vs_actual"] = plot_sim_vs_actual(
            df, symbol, Path(sim_csv),
            out_dir / f"{symbol}_sim_vs_actual.png"
        )

    # 5. Prediction + GARCH fit (if params provided)
    if params_path is not None and Path(params_path).exists():
        from stochax_market.predict import predict

        result      = predict(symbol=symbol, horizon=horizon,
                              params_path=params_path, seed=seed)
        print(type(result))
        print(result.keys() if isinstance(result, dict) else dir(result))
        print(result)

        last_date   = df["Date"].iloc[-1]
        pred_dates  = pd.bdate_range(last_date, periods=horizon + 1)[1:]

        saved["prediction"] = plot_prediction(
            df, symbol,
            pred_mean  = result["mean_prediction"],
            pred_lo    = result["lower_ci"],
            pred_hi    = result["upper_ci"],
            pred_dates = pred_dates,
            out        = out_dir / f"{symbol}_prediction_{horizon}d.png",
            recent_n   = recent_n,
        )
        saved["garch_fit"] = plot_garch_vol_fit(
            df, symbol, Path(params_path),
            out_dir / f"{symbol}_garch_fit.png"
        )

    return saved