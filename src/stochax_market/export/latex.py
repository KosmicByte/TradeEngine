"""Generate publication-quality LaTeX PDF documenting a fitted SPDE-GARCH model."""

from __future__ import annotations

import math
import pickle
import subprocess
import warnings
from pathlib import Path
from typing import Any

# Loss value below which calibration is considered converged.
_CONVERGENCE_LOSS_THRESHOLD = 1e6


def _fmt(value: Any, precision: int = 4) -> str:
    """Format a numeric value for LaTeX output.

    Returns ``\\text{undefined}`` for NaN/inf values, scientific notation for
    very small numbers, and fixed-point with *precision* decimal places otherwise.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return r"\text{undefined}"

    if math.isnan(v) or math.isinf(v):
        return r"\text{undefined}"

    if v != 0.0 and abs(v) < 1e-3:
        # Scientific notation for small values
        exp = int(math.floor(math.log10(abs(v))))
        mantissa = v / (10 ** exp)
        return f"{mantissa:.{precision}f} \\times 10^{{{exp}}}"

    return f"{v:.{precision}f}"


def _safe_sqrt(v: float) -> str:
    """Return formatted square root, or ``\\text{undefined}`` if non-positive."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return r"\text{undefined}"
    if math.isnan(v) or math.isinf(v) or v <= 0.0:
        return r"\text{undefined}"
    return _fmt(math.sqrt(v))


def _build_latex(
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
        loss_info: Dict with optional keys ``final_loss`` and ``steps``.

    Returns:
        Complete LaTeX document as a string.
    """
    # ------------------------------------------------------------------ #
    # Extract parameters via property accessors                           #
    # ------------------------------------------------------------------ #
    domain_L = float(spde.domain_extent)
    dt = float(spde.dt)
    nx = int(spde.nx)
    dx = domain_L / nx

    mu_scale = _fmt(float(spde.mu_scale))
    raw_mu_scale = _fmt(float(spde.raw_mu_scale))

    omega = _fmt(float(garch.omega))
    alpha = _fmt(float(garch.alpha))
    beta = _fmt(float(garch.beta))
    raw_omega = _fmt(float(garch.raw_omega))

    persistence_val = float(garch.alpha) + float(garch.beta)
    persistence = _fmt(persistence_val)

    denom = 1.0 - persistence_val
    uncond_var = float(garch.omega) / denom if denom > 0 else float("nan")
    uncond_vol = _safe_sqrt(uncond_var)

    final_loss = loss_info.get("final_loss")
    opt_steps = loss_info.get("steps")

    loss_str = _fmt(final_loss) if final_loss is not None else r"\text{N/A}"
    steps_str = str(int(opt_steps)) if opt_steps is not None else "N/A"
    converged = (
        "Converged"
        if (final_loss is not None and float(final_loss) < _CONVERGENCE_LOSS_THRESHOLD)
        else "Unknown"
    )

    domain_L_str = _fmt(domain_L)
    dt_str = _fmt(dt)
    dx_str = _fmt(dx)

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
Baseline variance & $\omega$ & ${omega}$ (raw: ${raw_omega}$) \\
ARCH coefficient  & $\alpha$ & ${alpha}$ (constrained: $[0.05,\,0.20]$) \\
GARCH coefficient & $\beta$  & ${beta}$ (constrained: $[0.50,\,0.90]$) \\
\midrule
Persistence       & $\alpha + \beta$ & ${persistence}$ \\
Unconditional vol & $\sqrt{{\omega/(1-\alpha-\beta)}}$ & ${uncond_vol}$ \\
\bottomrule
\end{{tabular}}
\caption{{GARCH(1,1) fitted parameters. Raw values refer to unconstrained
space before softplus/sigmoid transforms.}}
\end{{table}}

\subsection{{Stationarity Condition}}
The model is covariance-stationary since $\alpha + \beta = {persistence} < 1$.

% ============================================================
\section{{Drift Specification}}

The drift term is:
\begin{{equation}}
\text{{drift}}(t) = \mu_{{\text{{scale}}}} \cdot \log\left(\frac{{\text{{Close}}_t}}{{\text{{Close}}_{{t-1}}}}\right)
\end{{equation}}

\subsection{{Estimated Drift Parameter}}
\begin{{itemize}}
  \item $\mu_{{\text{{scale}}}} = {mu_scale}$ (transformed from
        $\text{{raw\_mu\_scale}} = {raw_mu_scale}$ via softplus)
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
& $\mu_{{\text{{scale}}}}$ (drift) & {mu_scale} \\
& $L$ (domain extent) & {domain_L_str} INR \\
& $n_x$ (grid points) & {nx} \\
& $\Delta t$ (time step) & 1/252 \\
\midrule
\multicolumn{{3}}{{l}}{{\textit{{GARCH(1,1)}}}} \\
& $\omega$ & {omega} \\
& $\alpha$ & {alpha} \\
& $\beta$ & {beta} \\
& $\alpha + \beta$ & {persistence} \\
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
    with open(params_path, "rb") as fh:
        data = pickle.load(fh)

    if isinstance(data, tuple):
        spde, garch, L = data
        loss_info: dict = {}
    elif isinstance(data, dict):
        spde = data["spde"]
        garch = data["garch"]
        L = data["L"]
        loss_info = data.get("loss_info", {})
    else:
        raise TypeError(
            f"Unrecognised params.pkl format: expected tuple or dict, got {type(data)!r}. "
            "Supported formats: (spde, garch, L) tuple or "
            "{'spde': ..., 'garch': ..., 'L': ..., 'loss_info': ...} dict."
        )

    # ------------------------------------------------------------------ #
    # Generate LaTeX source                                               #
    # ------------------------------------------------------------------ #
    latex_source = _build_latex(symbol, spde, garch, float(L), loss_info)

    tex_path.write_text(latex_source, encoding="utf-8")
    print(f"[✓] LaTeX source written to {tex_path}")

    # ------------------------------------------------------------------ #
    # Compile with pdflatex                                               #
    # ------------------------------------------------------------------ #
    try:
        result = subprocess.run(
            ["pdflatex", "--version"],
            capture_output=True,
            check=False,
        )
        pdflatex_available = result.returncode == 0
    except FileNotFoundError:
        pdflatex_available = False

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
