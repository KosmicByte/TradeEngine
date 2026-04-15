"""Tests for stochax_market.debug — unified debugging & inspection utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from stochax_market.debug import (
    display_jaxpr,
    display_math_eq,
    display_module,
    export_computation_graph,
    inject_debug_prints,
)
from stochax_market.model.spde import PIDController, SPDEStepper
from stochax_market.model.volatility import GARCHVolatility


# ══════════════════════════════════════════════════════════════
# display_module
# ══════════════════════════════════════════════════════════════

class TestDisplayModule:
    def test_returns_string(self):
        spde = SPDEStepper(nx=32)
        result = display_module(spde)
        assert isinstance(result, str)

    def test_contains_class_name(self):
        spde = SPDEStepper(nx=32)
        result = display_module(spde)
        assert "SPDEStepper" in result

    def test_trainable_label_present(self):
        spde = SPDEStepper(nx=32)
        result = display_module(spde)
        assert "[trainable]" in result

    def test_static_label_present(self):
        spde = SPDEStepper(nx=32)
        result = display_module(spde)
        assert "[static]" in result

    def test_garch_fields_listed(self):
        garch = GARCHVolatility()
        result = display_module(garch)
        assert "raw_omega" in result
        assert "raw_alpha" in result
        assert "raw_beta" in result

    def test_pid_submodule_traversed(self):
        pid = PIDController()
        result = display_module(pid)
        assert "raw_Kp" in result
        assert "raw_Ki" in result
        assert "raw_Kd" in result

    def test_plain_pytree(self):
        data = {"a": jnp.ones(3), "b": jnp.zeros((2, 2))}
        result = display_module(data)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_nested_module(self):
        spde = SPDEStepper(nx=32)
        result = display_module(spde)
        # diffusion_stepper is a nested Equinox Module inside SPDEStepper
        assert "diffusion_stepper" in result or "Diffusion" in result

    def test_array_shape_shown(self):
        garch = GARCHVolatility()
        result = display_module(garch)
        # scalar arrays show [] or shape info
        assert "Array" in result or "float32" in result


# ══════════════════════════════════════════════════════════════
# display_jaxpr
# ══════════════════════════════════════════════════════════════

class TestDisplayJaxpr:
    def _simple_fn(self, x, y):
        return x + y

    def test_returns_string(self):
        result = display_jaxpr(self._simple_fn, jnp.ones(3), jnp.ones(3))
        assert isinstance(result, str)

    def test_contains_jaxpr_header(self):
        result = display_jaxpr(self._simple_fn, jnp.ones(3), jnp.ones(3))
        assert "Jaxpr IR" in result

    def test_contains_primitive(self):
        result = display_jaxpr(self._simple_fn, jnp.ones(3), jnp.ones(3))
        assert "add" in result

    def test_input_shapes_shown(self):
        result = display_jaxpr(self._simple_fn, jnp.ones((4, 2)), jnp.ones((4, 2)))
        assert "4" in result

    def test_output_section_present(self):
        result = display_jaxpr(self._simple_fn, jnp.ones(3), jnp.ones(3))
        assert "Output" in result

    def test_equations_section_present(self):
        result = display_jaxpr(self._simple_fn, jnp.ones(3), jnp.ones(3))
        assert "Equation" in result

    def test_multiply_fn(self):
        result = display_jaxpr(lambda x: x * 2.0, jnp.ones(5))
        assert isinstance(result, str)
        assert "mul" in result or "integer_pow" in result or "broadcast" in result

    def test_kwargs_forwarded(self):
        def fn_with_kw(x, scale=1.0):
            return x * scale

        result = display_jaxpr(fn_with_kw, jnp.ones(3), scale=2.0)
        assert isinstance(result, str)
        assert "Jaxpr IR" in result


# ══════════════════════════════════════════════════════════════
# display_math_eq
# ══════════════════════════════════════════════════════════════

class TestDisplayMathEq:
    @pytest.mark.parametrize("name", [
        "spde", "spde_latex",
        "garch", "garch_latex",
        "loss", "loss_latex",
        "kl", "kl_latex",
        "field_mean", "field_mean_latex",
        "pid", "pid_latex",
    ])
    def test_known_names_return_string(self, name):
        result = display_math_eq(name)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_garch_eq_contains_sigma(self):
        eq = display_math_eq("garch")
        assert "σ" in eq or "sigma" in eq.lower()

    def test_spde_eq_contains_kappa(self):
        eq = display_math_eq("spde")
        assert "κ" in eq or "kappa" in eq.lower()

    def test_loss_eq_contains_mse(self):
        eq = display_math_eq("loss")
        assert "MSE" in eq or "mse" in eq.lower()

    def test_kl_eq_contains_sum(self):
        eq = display_math_eq("kl")
        assert "Σ" in eq or "sum" in eq.lower() or r"\sum" in eq

    def test_latex_variants_contain_backslash(self):
        eq = display_math_eq("spde_latex")
        assert "\\" in eq

    def test_unknown_name_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown equation"):
            display_math_eq("does_not_exist")

    def test_case_insensitive(self):
        assert display_math_eq("GARCH") == display_math_eq("garch")
        assert display_math_eq("SPDE") == display_math_eq("spde")

    def test_whitespace_stripped(self):
        assert display_math_eq("  garch  ") == display_math_eq("garch")


# ══════════════════════════════════════════════════════════════
# export_computation_graph
# ══════════════════════════════════════════════════════════════

class TestExportComputationGraph:
    @pytest.fixture
    def simple_jaxpr(self):
        return jax.make_jaxpr(lambda x: x ** 2)(jnp.ones(4))

    def test_text_format_returns_string(self, simple_jaxpr):
        result = export_computation_graph(simple_jaxpr, format="text")
        assert isinstance(result, str)

    def test_text_format_contains_inputs(self, simple_jaxpr):
        result = export_computation_graph(simple_jaxpr, format="text")
        assert "Input" in result

    def test_text_format_contains_outputs(self, simple_jaxpr):
        result = export_computation_graph(simple_jaxpr, format="text")
        assert "Output" in result

    def test_text_format_contains_equations(self, simple_jaxpr):
        result = export_computation_graph(simple_jaxpr, format="text")
        assert "Equation" in result

    def test_dot_format_is_digraph(self, simple_jaxpr):
        result = export_computation_graph(simple_jaxpr, format="dot")
        assert result.strip().startswith("digraph")
        assert "}" in result

    def test_dot_format_has_nodes(self, simple_jaxpr):
        result = export_computation_graph(simple_jaxpr, format="dot")
        assert "in_0" in result

    def test_html_format_has_html_tag(self, simple_jaxpr):
        result = export_computation_graph(simple_jaxpr, format="html")
        assert "<html>" in result or "<!DOCTYPE html>" in result

    def test_html_format_has_table(self, simple_jaxpr):
        result = export_computation_graph(simple_jaxpr, format="html")
        assert "<table>" in result

    def test_invalid_format_raises(self, simple_jaxpr):
        with pytest.raises(ValueError, match="Unknown format"):
            export_computation_graph(simple_jaxpr, format="pdf")

    def test_default_format_is_text(self, simple_jaxpr):
        result = export_computation_graph(simple_jaxpr)
        assert "Computation Graph (text)" in result

    def test_multi_op_function(self):
        def fn(x):
            return jnp.sin(x) + jnp.cos(x) * 2.0

        closed = jax.make_jaxpr(fn)(jnp.ones(8))
        result = export_computation_graph(closed, format="text")
        assert isinstance(result, str)
        assert len(result) > 0


# ══════════════════════════════════════════════════════════════
# inject_debug_prints
# ══════════════════════════════════════════════════════════════

class TestInjectDebugPrints:
    def test_returns_callable(self):
        fn = jax.jit(lambda x, y: x + y)
        wrapped = inject_debug_prints(fn, ["arg0"])
        assert callable(wrapped)

    def test_output_unchanged(self):
        @jax.jit
        def fn(x, y):
            return x + y

        wrapped = inject_debug_prints(fn, ["arg0", "arg1"])
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])
        result = wrapped(x, y)
        expected = fn(x, y)
        assert jnp.allclose(result, expected)

    def test_preserves_function_name(self):
        @jax.jit
        def my_loss(x):
            return jnp.sum(x)

        wrapped = inject_debug_prints(my_loss, ["arg0"])
        assert wrapped.__name__ == "my_loss"

    def test_attribute_key(self, capsys):
        """Attribute keys like 'arg0.raw_mu_scale' resolve correctly."""
        spde = SPDEStepper(nx=32)

        @jax.jit
        def identity(model):
            return model.raw_mu_scale

        wrapped = inject_debug_prints(identity, ["arg0.raw_mu_scale"])
        # Should not raise; output value is raw_mu_scale
        result = wrapped(spde)
        assert jnp.isfinite(result)

    def test_out_of_range_arg_key(self, capsys):
        """An arg index beyond the actual args count prints <not found>."""
        fn = jax.jit(lambda x: x * 2)
        wrapped = inject_debug_prints(fn, ["arg5"])
        # Should not raise
        result = wrapped(jnp.ones(3))
        assert jnp.allclose(result, jnp.full(3, 2.0))

    def test_non_jitted_function_wrapped(self):
        """inject_debug_prints works on non-JIT functions too."""
        def plain_fn(x):
            return x ** 2

        wrapped = inject_debug_prints(plain_fn, ["arg0"])
        result = wrapped(jnp.array([3.0]))
        assert jnp.allclose(result, jnp.array([9.0]))

    def test_empty_debug_keys(self):
        """Empty key list → wrapper passes through without printing."""
        fn = jax.jit(lambda x: x + 1.0)
        wrapped = inject_debug_prints(fn, [])
        result = wrapped(jnp.ones(4))
        assert jnp.allclose(result, jnp.full(4, 2.0))
