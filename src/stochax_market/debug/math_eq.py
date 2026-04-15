"""Mathematical equation display for the SPDE-GARCH calibration pipeline.

Provides a catalogue of LaTeX-like string representations for the core
mathematical entities used in *stochax-market*.  All values are pure constants
— no side effects beyond the return value.
"""

from __future__ import annotations

_EQUATIONS: dict[str, str] = {
    # Stochastic PDE
    "spde": (
        r"∂u/∂t = κ∇²u + μ(x,t)·u·drift(t) + σ(x,t)·u·Ẇ(x,t)"
    ),
    "spde_latex": (
        r"\frac{\partial u}{\partial t} = "
        r"\kappa \nabla^2 u "
        r"+ \mu_{\text{scale}} \cdot \text{drift}(t) \cdot u "
        r"+ \sigma(x,t) \cdot u \cdot \dot{W}(x,t)"
    ),
    # GARCH(1,1) recursion
    "garch": r"σ²_t = ω + α·X²_{t-1} + β·σ²_{t-1}",
    "garch_latex": (
        r"\sigma^2_t = \omega + \alpha X^2_{t-1} + \beta \sigma^2_{t-1}"
    ),
    # Calibration loss
    "loss": (
        r"ℒ = MSE(u(·,T)_mean, Close) + 0.1·directional_penalty"
    ),
    "loss_latex": (
        r"\mathcal{L}(\theta) = "
        r"\frac{1}{T}\sum_{t=1}^{T}\!\left(\hat{p}_t(\theta) - p_t\right)^2 "
        r"+ 0.1 \cdot \text{dir\_penalty}"
    ),
    # Karhunen-Loève expansion
    "kl": r"W(x) = Σᵢ √λᵢ ξᵢ φᵢ(x)",
    "kl_latex": (
        r"W(x) = \sum_{i=1}^{N} \sqrt{\lambda_i}\, \xi_i\, \varphi_i(x)"
    ),
    # Field-mean price extraction
    "field_mean": r"Price_t = L · ∫₀ᴸ x·u(t,x) dx",
    "field_mean_latex": (
        r"\text{Price}_t = L \int_0^L x\, u(t,x)\, \mathrm{d}x"
    ),
    # PID correction
    "pid": (
        r"u_PID(t) = Kp·e_t + Ki·∫e dt + Kd·(de/dt),  e_t = r_t - ū_t"
    ),
    "pid_latex": (
        r"u_{\text{PID}}(t) = K_p e_t + K_i \int e\,\mathrm{d}t "
        r"+ K_d \frac{\mathrm{d}e}{\mathrm{d}t}, \quad e_t = r_t - \bar{u}_t"
    ),
}

_KNOWN_NAMES = sorted(_EQUATIONS)


def display_math_eq(eq_name: str) -> str:
    """Return the LaTeX-like string for a named mathematical equation.

    The catalogue covers the principal entities used in the SPDE-GARCH
    calibration pipeline.

    Available names (plain / ``_latex`` variant):

    * ``"spde"`` / ``"spde_latex"``   — SPDE PDE form
    * ``"garch"`` / ``"garch_latex"`` — GARCH(1,1) recursion
    * ``"loss"`` / ``"loss_latex"``   — calibration loss
    * ``"kl"`` / ``"kl_latex"``       — Karhunen-Loève expansion
    * ``"field_mean"`` / ``"field_mean_latex"`` — price extraction
    * ``"pid"`` / ``"pid_latex"``     — PID correction

    Args:
        eq_name: Case-insensitive equation identifier.

    Returns:
        Equation string (Unicode or LaTeX math source).

    Raises:
        KeyError: If *eq_name* is not in the catalogue.  The error message
            lists all valid names.
    """
    key = eq_name.strip().lower()
    if key not in _EQUATIONS:
        raise KeyError(
            f"Unknown equation {eq_name!r}.  "
            f"Valid names: {_KNOWN_NAMES}"
        )
    return _EQUATIONS[key]
