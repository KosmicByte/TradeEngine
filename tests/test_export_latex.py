"""Tests for the LaTeX export module."""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stochax_market.export.latex import _build_latex, _fmt, _safe_sqrt, export_model_pdf
from stochax_market.model.spde import SPDEStepper
from stochax_market.model.volatility import GARCHVolatility


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_spde(domain_extent=100.0, dt=1.0 / 252.0, nx=128, mu_scale=0.1):
    """Return a minimal mock SPDEStepper with the required attributes."""
    spde = MagicMock()
    spde.domain_extent = domain_extent
    spde.dt = dt
    spde.nx = nx
    # softplus(inverse_softplus(mu_scale)) ≈ mu_scale
    spde.mu_scale = mu_scale
    spde.raw_mu_scale = math.log(math.exp(mu_scale) - 1.0)
    return spde


def _make_real_spde(domain_extent=2500.0, nx=64) -> SPDEStepper:
    return SPDEStepper(nx=nx, domain_extent=domain_extent)


def _make_real_garch() -> GARCHVolatility:
    return GARCHVolatility()


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestFmt:
    def test_normal_float(self):
        assert _fmt(1.23456) == "1.2346"

    def test_zero(self):
        assert _fmt(0.0) == "0.0000"

    def test_small_value(self):
        result = _fmt(1e-5)
        assert result == r"1.0000 \times 10^{-5}"

    def test_nan(self):
        assert _fmt(float("nan")) == r"\text{undefined}"

    def test_inf(self):
        assert _fmt(float("inf")) == r"\text{undefined}"

    def test_negative_inf(self):
        assert _fmt(float("-inf")) == r"\text{undefined}"


class TestSafeSqrt:
    def test_positive(self):
        result = _safe_sqrt(4.0)
        assert result == "2.0000"

    def test_zero(self):
        assert _safe_sqrt(0.0) == r"\text{undefined}"

    def test_negative(self):
        assert _safe_sqrt(-1.0) == r"\text{undefined}"

    def test_nan(self):
        assert _safe_sqrt(float("nan")) == r"\text{undefined}"


# ---------------------------------------------------------------------------
# Unit tests for _build_latex
# ---------------------------------------------------------------------------


class TestBuildLatex:
    def setup_method(self):
        self.spde = _make_mock_spde(domain_extent=2500.0, nx=128)
        self.garch = _make_real_garch()

    def test_returns_string(self):
        doc = _build_latex("RELIANCE", self.spde, self.garch, 2500.0, {})
        assert isinstance(doc, str)

    def test_contains_symbol(self):
        doc = _build_latex("RELIANCE", self.spde, self.garch, 2500.0, {})
        assert "RELIANCE" in doc

    def test_contains_all_sections(self):
        doc = _build_latex("TEST", self.spde, self.garch, 2500.0, {})
        assert r"\section{Stochastic Partial Differential Equation}" in doc
        assert r"\section{GARCH(1,1) Conditional Volatility}" in doc
        assert r"\section{Drift Specification}" in doc
        assert r"\section{Calibration Objective}" in doc
        assert r"\section{Price Forecast Extraction}" in doc
        assert r"\section{Complete Parameter Summary}" in doc

    def test_contains_loss_info(self):
        loss_info = {"final_loss": 0.001234, "steps": 100}
        doc = _build_latex("TEST", self.spde, self.garch, 2500.0, loss_info)
        assert "0.0012" in doc
        assert "100" in doc

    def test_loss_na_when_missing(self):
        doc = _build_latex("TEST", self.spde, self.garch, 2500.0, {})
        assert r"\text{N/A}" in doc

    def test_domain_extent_in_doc(self):
        doc = _build_latex("TEST", self.spde, self.garch, 2500.0, {})
        assert "2500.0000" in doc

    def test_garch_params_present(self):
        doc = _build_latex("TEST", self.spde, self.garch, 2500.0, {})
        # alpha should be in [0.05, 0.20]
        assert r"$\alpha$" in doc
        assert r"$\beta$" in doc
        assert r"$\omega$" in doc

    def test_begin_end_document(self):
        doc = _build_latex("TEST", self.spde, self.garch, 2500.0, {})
        assert r"\begin{document}" in doc
        assert r"\end{document}" in doc

    def test_booktabs_package(self):
        doc = _build_latex("TEST", self.spde, self.garch, 2500.0, {})
        assert r"\usepackage{booktabs}" in doc


# ---------------------------------------------------------------------------
# Integration tests for export_model_pdf
# ---------------------------------------------------------------------------


class TestExportModelPdf:
    def _write_params(self, tmp_path: Path, fmt: str = "tuple") -> Path:
        spde = _make_real_spde(domain_extent=2500.0, nx=64)
        garch = _make_real_garch()
        params_file = tmp_path / "params.pkl"
        if fmt == "tuple":
            payload = (spde, garch, 2500.0)
        else:
            payload = {
                "spde": spde,
                "garch": garch,
                "L": 2500.0,
                "loss_info": {"final_loss": 0.00123, "steps": 50},
            }
        with open(params_file, "wb") as fh:
            pickle.dump(payload, fh)
        return params_file

    def test_tuple_format_writes_tex(self, tmp_path):
        params_file = self._write_params(tmp_path, fmt="tuple")
        result = export_model_pdf("TEST", params_file, tmp_path / "out.pdf")
        # Either a .pdf or a .tex should exist
        assert result.exists()

    def test_dict_format_writes_tex(self, tmp_path):
        params_file = self._write_params(tmp_path, fmt="dict")
        result = export_model_pdf("TEST", params_file, tmp_path / "out.pdf")
        assert result.exists()

    def test_default_output_path(self, tmp_path):
        params_file = self._write_params(tmp_path, fmt="tuple")
        result = export_model_pdf("TEST", params_file)
        # Default should be next to params.pkl
        assert result.parent == tmp_path
        assert "TEST" in result.name

    def test_tex_file_created(self, tmp_path):
        params_file = self._write_params(tmp_path, fmt="tuple")
        export_model_pdf("TEST", params_file, tmp_path / "out.pdf")
        tex = tmp_path / "out.tex"
        assert tex.exists()
        content = tex.read_text()
        assert "TEST" in content

    def test_unknown_format_raises_type_error(self, tmp_path):
        bad_file = tmp_path / "bad.pkl"
        with open(bad_file, "wb") as fh:
            pickle.dump([1, 2, 3], fh)
        with pytest.raises(TypeError, match="Unrecognised params.pkl format"):
            export_model_pdf("TEST", bad_file, tmp_path / "out.pdf")

    def test_returns_path_object(self, tmp_path):
        params_file = self._write_params(tmp_path, fmt="dict")
        result = export_model_pdf("TEST", params_file, tmp_path / "out.pdf")
        assert isinstance(result, Path)
