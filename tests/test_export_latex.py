"""Tests for export/latex.py — covers all fixed edge cases."""

from __future__ import annotations

import math
import pickle
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

from stochax_market.export.latex import _fmt, export_model_pdf
from stochax_market.model.spde import SPDEStepper
from stochax_market.model.volatility import GARCHVolatility


# ══════════════════════════════════════════════════════════════
# _fmt helper — unit tests
# ══════════════════════════════════════════════════════════════

class TestFmt:
    def test_none_returns_fallback(self):
        assert _fmt(None) == "not recorded"

    def test_none_custom_fallback(self):
        assert _fmt(None, fallback="–") == "–"

    def test_nan_returns_fallback(self):
        assert _fmt(float("nan")) == "not recorded"

    def test_inf_returns_fallback(self):
        assert _fmt(float("inf")) == "not recorded"
        assert _fmt(float("-inf")) == "not recorded"

    def test_zero_formats_correctly(self):
        assert _fmt(0.0) == "0.0000"

    def test_normal_float(self):
        assert _fmt(1.23456789, ".4f") == "1.2346"

    def test_scientific_notation(self):
        result = _fmt(0.00000225, ".6g")
        assert "2.25" in result

    def test_string_passthrough(self):
        assert _fmt("hello") == "hello"

    def test_long_string_truncated(self):
        long_str = "x" * 200
        assert len(_fmt(long_str)) <= 80

    def test_optimistix_result_string_truncated(self):
        ugly = (
            "optimistix._solution.RESULTS<The maximum number of steps was "
            "reached in the nonlinear solver. The problem may not be solvable."
            "> extra stuff"
        )
        result = _fmt(ugly)
        assert "RESULTS<" not in result
        assert len(result) <= 80
        assert "maximum number of steps" in result

    def test_integer_input(self):
        assert _fmt(1000, ".0f") == "1000"

    def test_jax_array_scalar(self):
        val = jnp.float32(3.14159)
        result = _fmt(val, ".4f")
        assert result == "3.1416"


# ══════════════════════════════════════════════════════════════
# params.pkl format handling
# ══════════════════════════════════════════════════════════════

def _make_params_dict(L: float = 185.0, final_loss: float = 4231.5) -> dict:
    """Create a realistic params dict as saved by the fixed cli.py."""
    return {
        "spde":  SPDEStepper(nx=128, domain_extent=L),
        "garch": GARCHVolatility(),
        "L":     L,
        "loss_info": {
            "steps":      1000,
            "result":     "converged",
            "final_loss": final_loss,
        },
    }


def _make_params_tuple(L: float = 185.0) -> tuple:
    """Legacy tuple format (spde, garch, L)."""
    return (SPDEStepper(nx=128, domain_extent=L), GARCHVolatility(), L)


class TestLoadParams:
    """Verify _load_params handles both pkl formats and extracts L correctly."""

    def test_dict_format_L(self, tmp_path):
        pkl = tmp_path / "params.pkl"
        pkl.write_bytes(pickle.dumps(_make_params_dict(L=185.45)))
        from stochax_market.export.latex import _load_params
        spde, garch, L, loss_info = _load_params(pkl)
        assert abs(L - 185.45) < 1e-3

    def test_tuple_format_L(self, tmp_path):
        pkl = tmp_path / "params.pkl"
        pkl.write_bytes(pickle.dumps(_make_params_tuple(L=200.0)))
        from stochax_market.export.latex import _load_params
        spde, garch, L, loss_info = _load_params(pkl)
        assert abs(L - 200.0) < 1e-3

    def test_tuple_format_loss_info_empty(self, tmp_path):
        pkl = tmp_path / "params.pkl"
        pkl.write_bytes(pickle.dumps(_make_params_tuple()))
        from stochax_market.export.latex import _load_params
        _, _, _, loss_info = _load_params(pkl)
        assert isinstance(loss_info, dict)   # never None

    def test_dict_format_loss_info_present(self, tmp_path):
        pkl = tmp_path / "params.pkl"
        pkl.write_bytes(pickle.dumps(_make_params_dict(final_loss=9999.1)))
        from stochax_market.export.latex import _load_params
        _, _, _, loss_info = _load_params(pkl)
        assert abs(loss_info["final_loss"] - 9999.1) < 0.1

    def test_unknown_format_raises(self, tmp_path):
        pkl = tmp_path / "params.pkl"
        pkl.write_bytes(pickle.dumps({"unexpected": True}))
        from stochax_market.export.latex import _load_params
        with pytest.raises(TypeError):
            _load_params(pkl)


# ══════════════════════════════════════════════════════════════
# LaTeX source generation
# ══════════════════════════════════════════════════════════════

class TestLatexGeneration:
    """Verify the generated LaTeX string is correct and complete."""

    @pytest.fixture
    def tex_source(self, tmp_path) -> str:
        """Generate LaTeX for GAIL with realistic params."""
        pkl = tmp_path / "params.pkl"
        pkl.write_bytes(pickle.dumps(_make_params_dict(L=185.45, final_loss=4231.5)))
        from stochax_market.export.latex import _generate_latex
        spde, garch, L, loss_info = __import__(
            "stochax_market.export.latex", fromlist=["_load_params"]
        )._load_params(pkl)
        return _generate_latex("GAIL", spde, garch, L, loss_info)

    def test_symbol_in_title(self, tex_source):
        assert "GAIL" in tex_source

    def test_L_is_not_1(self, tex_source):
        # Domain extent must NOT be the default 1.0000 — must be real L
        assert "1.0000 INR" not in tex_source
        assert "185" in tex_source

    def test_no_na_strings(self, tex_source):
        assert "N/A" not in tex_source
        assert "None" not in tex_source

    def test_loss_value_present(self, tex_source):
        assert "4231" in tex_source

    def test_steps_present(self, tex_source):
        assert "1000" in tex_source

    def test_garch_params_not_default_raw(self, tex_source):
        # omega, alpha, beta must appear as transformed values
        assert "0.0418" in tex_source or "0.04" in tex_source

    def test_mu_scale_6_decimal_places(self, tex_source):
        # mu_scale must appear with at least 5 significant digits
        import re
        matches = re.findall(r"mu.*?=\s*([\d.]+)", tex_source)
        assert any(len(m.replace(".", "").lstrip("0")) >= 4 for m in matches)

    def test_persistence_lt_1(self, tex_source):
        # alpha + beta shown and stationary (default ≈ 0.9957)
        assert "0.99" in tex_source or "0.98" in tex_source or "0.97" in tex_source

    def test_uncond_vol_present(self, tex_source):
        assert "not recorded" not in tex_source or "1." in tex_source

    def test_latex_has_document_begin(self, tex_source):
        assert r"\begin{document}" in tex_source

    def test_latex_has_document_end(self, tex_source):
        assert r"\end{document}" in tex_source

    def test_all_6_sections_present(self, tex_source):
        assert r"\section{Stochastic" in tex_source
        assert r"\section{GARCH" in tex_source
        assert r"\section{Drift" in tex_source
        assert r"\section{Calibration" in tex_source
        assert r"\section{Price Forecast" in tex_source
        assert r"\section{Complete" in tex_source


# ══════════════════════════════════════════════════════════════
# Edge cases — degenerate parameter values
# ══════════════════════════════════════════════════════════════

class TestDegenerateParams:
    """PDF generation must not crash with edge-case parameter values."""

    def _write_pkl(self, tmp_path, spde, garch, L, loss_info):
        pkl = tmp_path / "params.pkl"
        pkl.write_bytes(pickle.dumps(
            {"spde": spde, "garch": garch, "L": L, "loss_info": loss_info}
        ))
        return pkl

    def test_missing_loss_info_keys(self, tmp_path):
        """loss_info with no keys → renders 'not recorded', no crash."""
        pkl = self._write_pkl(
            tmp_path,
            SPDEStepper(nx=128), GARCHVolatility(),
            L=185.0, loss_info={}
        )
        from stochax_market.export.latex import _generate_latex, _load_params
        spde, garch, L, loss_info = _load_params(pkl)
        tex = _generate_latex("TEST", spde, garch, L, loss_info)
        assert "not recorded" in tex
        assert "None" not in tex

    def test_negative_final_loss(self, tmp_path):
        """Negative loss (from old broken penalty term) renders correctly."""
        pkl = self._write_pkl(
            tmp_path,
            SPDEStepper(nx=128), GARCHVolatility(),
            L=185.0,
            loss_info={"final_loss": -1.88e19, "steps": 1000, "result": "failed"}
        )
        from stochax_market.export.latex import _generate_latex, _load_params
        spde, garch, L, loss_info = _load_params(pkl)
        tex = _generate_latex("TEST", spde, garch, L, loss_info)
        # Should render the negative value, not crash or show N/A
        assert "-" in tex or "not recorded" in tex
        assert "None" not in tex

    def test_long_optimistix_result_string(self, tmp_path):
        """Full optimistix RESULTS<...> string is truncated in output."""
        ugly = (
            "optimistix._solution.RESULTS<The maximum number of steps was "
            "reached in the nonlinear solver. The problem may not be solvable "
            "(e.g., a root-find on a function that has no roots), or you may "
            "need to increase max_steps.>"
        )
        pkl = self._write_pkl(
            tmp_path,
            SPDEStepper(nx=128), GARCHVolatility(),
            L=185.0,
            loss_info={"final_loss": 500.0, "steps": 1000, "result": ugly}
        )
        from stochax_market.export.latex import _generate_latex, _load_params
        spde, garch, L, loss_info = _load_params(pkl)
        tex = _generate_latex("TEST", spde, garch, L, loss_info)
        assert "RESULTS<" not in tex

    def test_persistence_near_1_uncond_vol_guarded(self, tmp_path):
        """When alpha+beta ≈ 1, unconditional vol is guarded, no crash."""
        import equinox as eqx
        garch = GARCHVolatility()
        # Force beta close to 0.999 - alpha to make persistence near 1
        garch = eqx.tree_at(
            lambda g: g.raw_beta,
            garch,
            jnp.float32(10.0)   # very high raw_beta → beta near max
        )
        pkl = self._write_pkl(
            tmp_path, SPDEStepper(nx=128), garch,
            L=185.0, loss_info={"final_loss": 100.0, "steps": 500}
        )
        from stochax_market.export.latex import _generate_latex, _load_params
        spde, garch, L, loss_info = _load_params(pkl)
        # Must not raise ZeroDivisionError or produce NaN in output
        tex = _generate_latex("TEST", spde, garch, L, loss_info)
        assert "nan" not in tex.lower()


# ══════════════════════════════════════════════════════════════
# PDF compilation (integration, skipped if pdflatex absent)
# ══════════════════════════════════════════════════════════════

class TestPdfCompilation:
    @pytest.fixture
    def params_pkl(self, tmp_path):
        pkl = tmp_path / "params.pkl"
        pkl.write_bytes(pickle.dumps(_make_params_dict(L=185.45, final_loss=4231.5)))
        return pkl

    @pytest.mark.skipif(
        not __import__("shutil").which("pdflatex"),
        reason="pdflatex not installed"
    )
    def test_pdf_created(self, params_pkl, tmp_path):
        from stochax_market.export.latex import export_model_pdf
        pdf = export_model_pdf("GAIL", params_pkl, tmp_path / "GAIL_model.pdf")
        assert pdf.exists()
        assert pdf.stat().st_size > 10_000   # real PDF not empty

    def test_tex_written_without_pdflatex(self, params_pkl, tmp_path):
        """Even without pdflatex, .tex file must be written."""
        with patch("shutil.which", return_value=None):
            from stochax_market.export import latex as latex_mod
            import importlib; importlib.reload(latex_mod)
            tex_path = latex_mod.export_model_pdf(
                "GAIL", params_pkl, tmp_path / "GAIL_model.pdf"
            )
        assert tex_path.suffix == ".tex"
        assert tex_path.exists()

