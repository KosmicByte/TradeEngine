"""Generate publication-quality LaTeX PDF documenting a fitted SPDE-GARCH model."""

from __future__ import annotations

import math
import pickle
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any


def _fmt(val: Any, spec: str = ".4f", fallback: str = "not recorded") -> str:
    """Format a numeric or string value safely for LaTeX.

    Returns fallback string if val is None, NaN, inf, or unformattable.
    Truncates long strings to 80 characters to prevent LaTeX table overflow.

    Args:
        val:      Value to format (numeric or str).
        spec:     Python format spec for numeric values (default ".4f").
        fallback: String returned when val cannot be formatted.

    Returns:
        LaTeX-safe formatted string.
    """
    if val is None:
        return fallback
    if isinstance(val, str):
        if "RESULTS<" in val:
            try:
                inner = val.split("<", 1)[1].rstrip(">")
                val = inner.split(".")[0].strip()
            except (IndexError, AttributeError):
                pass
        return val[:80]
    try:
        v = float(val)
        if v != v or abs(v) == float("inf"):
            return fallback
        return format(v, spec)
    except (TypeError, ValueError):
        return fallback


def _safe_sqrt(v: float) -> str:
    """Return formatted square root, or ``\\text{undefined}`` if non-positive."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return r"\text{undefined}"
    if math.isnan(v) or math.isinf(v) or v <= 0.0:
        return r"\text{undefined}"
    return _fmt(math.sqrt(v))


def _generate_latex(
    symbol: str,
    spde: Any,
    garch: Any,
    L: float,
    loss_info: dict,
) -> str:
    """Build the complete LaTeX document string.

    Args:
        symbol: Stock symbol string.
        spde: Fitted :class:`~stochax_market.model.spde.SPDEStepper` instance.
        garch: Fitted :class:`~stochax_market.model.volatility.GARCHVolatility`
            instance.
        L: Domain extent (from features or spde.domain_extent).
        loss_info: Dict with optional keys ``final_loss``, ``steps``, and
            ``result``.

    Returns:
        Complete LaTeX document as a string.
    """
    # ------------------------------------------------------------------ #
    # Extract parameters via property accessors                           #
    # ------------------------------------------------------------------ #
    dt = float(spde.dt)
    nx = int(spde.nx)
    dx = L / nx

    mu_scale = float(spde.mu_scale)
    raw_mu_scale = float(spde.raw_mu_scale)

    omega = float(garch.omega)
    alpha = float(garch.alpha)
    beta = float(garch.beta)
    persistence = alpha + beta

    # Unconditional vol: sqrt(omega / (1 - alpha - beta))
    # Guard: denominator may be <= 0 if persistence >= 1.
    # Use 1e-6 tolerance to handle floating-point values numerically close to 1.
    denom = 1.0 - persistence
    if denom > 1e-6:
        uncond_vol = (omega / denom) ** 0.5
    else:
        uncond_vol = float("inf")   # _fmt will render as "not recorded"

    omega_str       = _fmt(omega,       ".4f")
    alpha_str       = _fmt(alpha,       ".4f")
    beta_str        = _fmt(beta,        ".4f")
    raw_omega_str   = _fmt(float(garch.raw_omega), ".4f")
    persistence_str = _fmt(persistence, ".4f")
    uncond_vol_str  = _fmt(uncond_vol,  ".4f")

    mu_scale_str     = _fmt(mu_scale,     ".6f")
    raw_mu_scale_str = _fmt(raw_mu_scale, ".4f")

    domain_L_str = _fmt(L,  ".4f")
    dt_str       = _fmt(dt, ".4f")
    dx_str       = _fmt(dx, ".4f")

    loss_str   = _fmt(loss_info.get("final_loss"), ".6g")
    steps_str  = _fmt(loss_info.get("steps"),      ".0f")
    converged  = _fmt(loss_info.get("result"),     fallback="not recorded")

    # ------------------------------------------------------------------ #
    # Build LaTeX source                                                  #
    # ------------------------------------------------------------------ #
    doc = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage{{amsmath,amssymb,bm}}
\usepackage{{geometry}}
\geometry{{margin=2.5cm}}
\usepackage{{booktabs}}
\usepackage{{xcolor}}
\definecolor{{accent}}{{RGB}}{{91,141,239}}

\title{{\textbf{{SPDE-GARCH Model Specification}}\\[0.3em]\large Stock: \texttt{{{symbol}}}}}
\author{{Calibrated via Optimistix BFGS}}
\date{{\today}}

\begin{{document}}
\maketitle

% ============================================================
\section{{Stochastic Partial Differential Equation}}

The price field $u(t,x)$ evolves according to:

\begin{{equation}}
\frac{{\partial u}}{{\partial t}} = \kappa \frac{{\partial^2 u}}{{\partial x^2}}
  + \mu \, \text{{drift}}(t) \, u
  + \sigma(t,x) \, u \, \frac{{\partial W}}{{\partial t}}
\end{{equation}}

where:
\begin{{itemize}}
  \item $\kappa = 0.01$ is the diffusion coefficient (fixed),
  \item $\mu = \mu_{{\text{{scale}}}}$ controls drift magnitude,
  \item $\sigma(t,x)$ is the spatial volatility field from GARCH(1,1),
  \item $W(t,x)$ is a $Q$-Wiener process on $[0,L]$.
\end{{itemize}}

\subsection{{Spatial Domain \& Discretization}}
\begin{{itemize}}
  \item Domain extent: $L = {domain_L_str}$ INR
  \item Grid points: $n_x = {nx}$
  \item Spatial resolution: $\Delta x = L / n_x = {dx_str}$ INR
  \item Time step: $\Delta t = {dt_str} \approx 1/252$ (daily)
\end{{itemize}}

\subsection{{Noise Process}}
The $Q$-Wiener increment is constructed via Karhunen-Lo\`{{e}}ve expansion:
\begin{{equation}}
W(x) = \sum_{{i=1}}^{{32}} \sqrt{{\lambda_i}} \, \xi_i \, \phi_i(x)
\end{{equation}}
with eigenvalues $\lambda_i = (0.015 / i)^2$ and eigenfunctions
$\phi_i(x) = \sqrt{{2/L}} \sin(i \pi x / L)$.

% ============================================================
\section{{GARCH(1,1) Conditional Volatility}}

The temporal volatility $\sigma_t$ follows:
\begin{{equation}}
\sigma_t^2 = \omega + \alpha \, X_{{t-1}}^2 + \beta \, \sigma_{{t-1}}^2
\end{{equation}}

where $X_t$ are log-returns. The spatial field is constructed as:
\begin{{equation}}
\sigma(t,x) = \sigma_t \cdot n_x \cdot \left(1 + 0.1 \sin\left(\frac{{\pi x}}{{L}}\right)\right)
\end{{equation}}

\subsection{{Estimated Parameters}}
\begin{{table}}[h]
\centering
\begin{{tabular}}{{lcc}}
\toprule
Parameter & Symbol & Fitted Value \\
\midrule
Baseline variance & $\omega$ & ${omega_str}$ (raw: ${raw_omega_str}$) \\
ARCH coefficient  & $\alpha$ & ${alpha_str}$ (constrained: $[0.05,\,0.20]$) \\
GARCH coefficient & $\beta$  & ${beta_str}$ (constrained: $[0.50,\,0.90]$) \\
\midrule
Persistence       & $\alpha + \beta$ & ${persistence_str}$ \\
Unconditional vol & $\sqrt{{\omega/(1-\alpha-\beta)}}$ & ${uncond_vol_str}$ \\
\bottomrule
\end{{tabular}}
\caption{{GARCH(1,1) fitted parameters. Raw values refer to unconstrained
space before softplus/sigmoid transforms.}}
\end{{table}}

\subsection{{Stationarity Condition}}
The model is covariance-stationary since $\alpha + \beta = {persistence_str} < 1$.

% ============================================================
\section{{Drift Specification}}

The drift term is:
\begin{{equation}}
\text{{drift}}(t) = \mu_{{\text{{scale}}}} \cdot \log\left(\frac{{\text{{Close}}_t}}{{\text{{Close}}_{{t-1}}}}\right)
\end{{equation}}

\subsection{{Estimated Drift Parameter}}
\begin{{itemize}}
  \item $\mu_{{\text{{scale}}}} = {mu_scale_str}$ (transformed from
        $\text{{raw\_mu\_scale}} = {raw_mu_scale_str}$ via softplus)
\end{{itemize}}

% ============================================================
\section{{Calibration Objective}}

The model was fitted by minimising the mean squared error between predicted
and actual closing prices:
\begin{{equation}}
\mathcal{{L}}(\theta) = \frac{{1}}{{T}} \sum_{{t=1}}^{{T}}
  \left(\text{{Pred}}_t(\theta) - \text{{Close}}_t\right)^2
\end{{equation}}

where $\theta = \{{\mu_{{\text{{scale}}}},\,\omega,\,\alpha,\,\beta\}}$ are the
trainable parameters.

\subsection{{Optimization Results}}
\begin{{itemize}}
  \item Solver: BFGS (Optimistix)
  \item Final loss: $\mathcal{{L}}^* = {loss_str}$
  \item Optimization steps: {steps_str}
  \item Convergence status: {converged}
\end{{itemize}}

% ============================================================
\section{{Price Forecast Extraction}}

At each timestep $t$, the predicted price is the first moment of the field:
\begin{{equation}}
\text{{Price}}_t = L \cdot \int_0^L x \, u(t,x) \, dx
\end{{equation}}

This is computed via trapezoidal quadrature on the grid $\{{x_i\}}_{{i=1}}^{{n_x}}$.

% ============================================================
\section{{Complete Parameter Summary}}

\begin{{table}}[h]
\centering
\begin{{tabular}}{{lcl}}
\toprule
\textbf{{Component}} & \textbf{{Parameter}} & \textbf{{Value}} \\
\midrule
\multicolumn{{3}}{{l}}{{\textit{{SPDE}}}} \\
& $\kappa$ (diffusion) & 0.01 \\
& $\mu_{{\text{{scale}}}}$ (drift) & {mu_scale_str} \\
& $L$ (domain extent) & {domain_L_str} INR \\
& $n_x$ (grid points) & {nx} \\
& $\Delta t$ (time step) & 1/252 \\
\midrule
\multicolumn{{3}}{{l}}{{\textit{{GARCH(1,1)}}}} \\
& $\omega$ & {omega_str} \\
& $\alpha$ & {alpha_str} \\
& $\beta$ & {beta_str} \\
& $\alpha + \beta$ & {persistence_str} \\
\midrule
\multicolumn{{3}}{{l}}{{\textit{{Noise (KL expansion)}}}} \\
& modes & 32 \\
& $\lambda_1$ & $(0.015)^2 = 0.000225$ \\
\midrule
\multicolumn{{3}}{{l}}{{\textit{{Calibration}}}} \\
& Final MSE loss & {loss_str} \\
& BFGS steps & {steps_str} \\
\bottomrule
\end{{tabular}}
\caption{{Complete model specification for \texttt{{{symbol}}}.}}
\end{{table}}

\end{{document}}
"""
    return doc


# Backward-compatible alias
_build_latex = _generate_latex


def _load_params(params_path: str | Path) -> tuple:
    """Load and unpack a ``params.pkl`` file.

    Supports both the legacy tuple format ``(spde, garch, L)`` and the
    dict format ``{"spde": ..., "garch": ..., "L": ..., "loss_info": ...}``.

    Args:
        params_path: Path to ``params.pkl``.

    Returns:
        Tuple ``(spde, garch, L, loss_info)`` where ``loss_info`` is always
        a dict (empty when not stored).

    Raises:
        TypeError: If the format is not recognised.
    """
    params_path = Path(params_path)
    with open(params_path, "rb") as fh:
        data = pickle.load(fh)

    if isinstance(data, tuple):
        spde, garch, L = data
        loss_info: dict = {}
    elif isinstance(data, dict) and "spde" in data and "garch" in data:
        spde = data["spde"]
        garch = data["garch"]
        L = data["L"]
        loss_info = data.get("loss_info") or {}
    else:
        raise TypeError(
            f"Unrecognised params.pkl format: expected tuple or dict, got {type(data)!r}. "
            "Supported formats: (spde, garch, L) tuple or "
            "{'spde': ..., 'garch': ..., 'L': ..., 'loss_info': ...} dict."
        )

    return spde, garch, float(L), loss_info


def export_model_pdf(
    symbol: str,
    params_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Load fitted params.pkl and generate LaTeX PDF with complete model spec.

    Args:
        symbol: Stock symbol (e.g. ``"RELIANCE"``).
        params_path: Path to fitted ``params.pkl``.
        output_path: Optional PDF output path (default: ``{symbol}_model.pdf``
            in the same directory as ``params_path``).

    Returns:
        :class:`~pathlib.Path` to the generated PDF (or ``.tex`` file if
        ``pdflatex`` is not available).

    Raises:
        TypeError: If the ``params.pkl`` format is not recognised.
    """
    params_path = Path(params_path)

    # ------------------------------------------------------------------ #
    # Determine output paths                                              #
    # ------------------------------------------------------------------ #
    if output_path is None:
        output_path = params_path.parent / f"{symbol}_model.pdf"
    output_path = Path(output_path)

    tex_path = output_path.with_suffix(".tex")

    # ------------------------------------------------------------------ #
    # Load params.pkl                                                     #
    # ------------------------------------------------------------------ #
    spde, garch, L, loss_info = _load_params(params_path)

    # ------------------------------------------------------------------ #
    # Generate LaTeX source                                               #
    # ------------------------------------------------------------------ #
    latex_source = _generate_latex(symbol, spde, garch, L, loss_info)

    tex_path.write_text(latex_source, encoding="utf-8")
    print(f"[✓] LaTeX source written to {tex_path}")

    # ------------------------------------------------------------------ #
    # Compile with pdflatex                                               #
    # ------------------------------------------------------------------ #
    pdflatex_available = shutil.which("pdflatex") is not None

    if not pdflatex_available:
        warnings.warn(
            f"pdflatex not found. LaTeX source saved to {tex_path}. "
            "Install TeX Live to generate PDF: sudo apt install texlive",
            stacklevel=2,
        )
        return tex_path

    print("[✓] Compiling with pdflatex...")

    # Run twice so cross-references / TOC are resolved
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=tex_path.parent,
            capture_output=True,
            check=False,
        )

    # ------------------------------------------------------------------ #
    # Clean up auxiliary files                                            #
    # ------------------------------------------------------------------ #
    pdf_path = output_path.with_suffix(".pdf")

    if not pdf_path.exists():
        # Keep the .log so the caller can diagnose compile errors.
        for ext in (".aux", ".out", ".toc"):
            aux = output_path.with_suffix(ext)
            if aux.exists():
                aux.unlink()
        log_path = output_path.with_suffix(".log")
        warnings.warn(
            f"pdflatex ran but {pdf_path} was not produced. "
            f"Check {log_path} for errors.",
            stacklevel=2,
        )
        return tex_path

    for ext in (".aux", ".log", ".out", ".toc"):
        aux = output_path.with_suffix(ext)
        if aux.exists():
            aux.unlink()

    size_kb = pdf_path.stat().st_size // 1024
    print(f"[✓] Model exported to {pdf_path} ({size_kb} KB)")
    return pdf_path
