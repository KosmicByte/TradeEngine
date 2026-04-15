"""JAX IR inspection and computation-graph export utilities.

``display_jaxpr`` traces a function and returns a formatted string of its
Jaxpr (JAX intermediate representation).

``export_computation_graph`` serialises the traced Jaxpr to *text*, *dot*
(Graphviz), or *html* format for external visualization.
"""

from __future__ import annotations

import html
import io
import textwrap
from collections.abc import Callable
from typing import Any

import jax


def display_jaxpr(fn: Callable, *args: Any, **kwargs: Any) -> str:
    """Return the Jaxpr IR of *fn* evaluated on concrete example *args*.

    Calls :func:`jax.make_jaxpr` to trace the function without executing it,
    then formats the resulting closed Jaxpr as a human-readable string that
    shows XLA primitives, input/output abstract shapes, and the equation DAG.

    Args:
        fn: A JAX-compatible callable (must be traceable by
            :func:`jax.make_jaxpr`).
        *args: Concrete example inputs used for tracing.  Their shapes and
            dtypes determine the abstract values in the Jaxpr.
        **kwargs: Additional keyword arguments forwarded to *fn* during
            tracing.

    Returns:
        Formatted string containing the full Jaxpr with one equation per line.

    Example::

        import jax.numpy as jnp
        from stochax_market.debug import display_jaxpr

        def add(x, y):
            return x + y

        print(display_jaxpr(add, jnp.ones(3), jnp.ones(3)))
    """
    closed_jaxpr = jax.make_jaxpr(lambda *a: fn(*a, **kwargs))(*args)

    buf = io.StringIO()
    buf.write("=== Jaxpr IR ===\n")
    buf.write(f"fn: {getattr(fn, '__name__', repr(fn))}\n")
    buf.write("\n--- Input shapes ---\n")
    for i, var in enumerate(closed_jaxpr.jaxpr.invars):
        buf.write(f"  in_{i}: {var.aval}\n")
    buf.write("\n--- Output shapes ---\n")
    for i, var in enumerate(closed_jaxpr.jaxpr.outvars):
        aval = var.aval if hasattr(var, "aval") else var
        buf.write(f"  out_{i}: {aval}\n")
    buf.write("\n--- Equations ---\n")
    for eq in closed_jaxpr.jaxpr.eqns:
        lhs = ", ".join(str(v) for v in eq.outvars)
        rhs_args = ", ".join(str(v) for v in eq.invars)
        params = (
            "  params=" + str(eq.params)
            if eq.params
            else ""
        )
        buf.write(f"  {lhs} = {eq.primitive.name}({rhs_args}){params}\n")
    buf.write("\n--- Constants ---\n")
    for i, const in enumerate(closed_jaxpr.consts):
        buf.write(f"  const_{i}: {const}\n")
    return buf.getvalue()


def _jaxpr_to_dot(closed_jaxpr: Any) -> str:
    """Convert a ClosedJaxpr to a Graphviz DOT string."""
    lines = [
        "digraph jaxpr {",
        '  rankdir=LR;',
        '  node [shape=box, style=filled, fillcolor="#e8f4f8"];',
    ]

    # Input nodes
    for i, var in enumerate(closed_jaxpr.jaxpr.invars):
        label = html.escape(f"in_{i}\\n{var.aval}")
        lines.append(
            f'  "in_{i}" [label="{label}", fillcolor="#c8e6c9"];'
        )

    # Equation nodes
    for eq_i, eq in enumerate(closed_jaxpr.jaxpr.eqns):
        label = html.escape(str(eq.primitive.name))
        if eq.params:
            param_str = ", ".join(f"{k}={v}" for k, v in eq.params.items())
            label += "\\n" + html.escape(param_str[:60])
        lines.append(f'  "eq_{eq_i}" [label="{label}"];')

    # Output nodes
    for i, var in enumerate(closed_jaxpr.jaxpr.outvars):
        aval = var.aval if hasattr(var, "aval") else var
        label = html.escape(f"out_{i}\\n{aval}")
        lines.append(
            f'  "out_{i}" [label="{label}", fillcolor="#ffccbc"];'
        )

    # Build a var -> producer map
    var_producer: dict[str, str] = {}
    for i, var in enumerate(closed_jaxpr.jaxpr.invars):
        var_producer[str(var)] = f"in_{i}"

    for eq_i, eq in enumerate(closed_jaxpr.jaxpr.eqns):
        for invar in eq.invars:
            src = var_producer.get(str(invar))
            if src is not None:
                lines.append(f'  "{src}" -> "eq_{eq_i}";')
        for outvar in eq.outvars:
            var_producer[str(outvar)] = f"eq_{eq_i}"

    for i, var in enumerate(closed_jaxpr.jaxpr.outvars):
        src = var_producer.get(str(var))
        if src is not None:
            lines.append(f'  "{src}" -> "out_{i}";')

    lines.append("}")
    return "\n".join(lines)


def _jaxpr_to_html(closed_jaxpr: Any) -> str:
    """Convert a ClosedJaxpr to a self-contained HTML page."""
    rows = []
    for i, var in enumerate(closed_jaxpr.jaxpr.invars):
        rows.append(
            f"<tr><td>in_{i}</td><td>input</td><td></td>"
            f"<td>{html.escape(str(var.aval))}</td></tr>"
        )
    for eq_i, eq in enumerate(closed_jaxpr.jaxpr.eqns):
        invars = html.escape(", ".join(str(v) for v in eq.invars))
        outvars = html.escape(", ".join(str(v) for v in eq.outvars))
        params = html.escape(str(eq.params) if eq.params else "")
        rows.append(
            f"<tr><td>{outvars}</td>"
            f"<td>{html.escape(eq.primitive.name)}</td>"
            f"<td>{invars}</td>"
            f"<td>{params}</td></tr>"
        )
    for i, var in enumerate(closed_jaxpr.jaxpr.outvars):
        aval = var.aval if hasattr(var, "aval") else var
        rows.append(
            f"<tr><td>out_{i}</td><td>output</td><td>{html.escape(str(var))}</td>"
            f"<td>{html.escape(str(aval))}</td></tr>"
        )

    table_body = "\n".join(rows)
    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>Jaxpr Computation Graph</title>
          <style>
            body {{ font-family: monospace; padding: 1em; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
            th {{ background: #4a90d9; color: white; }}
            tr:nth-child(even) {{ background: #f4f4f4; }}
          </style>
        </head>
        <body>
          <h2>Jaxpr Computation Graph</h2>
          <table>
            <thead>
              <tr><th>Output var</th><th>Primitive</th><th>Input vars</th><th>Params / shape</th></tr>
            </thead>
            <tbody>
        {table_body}
            </tbody>
          </table>
        </body>
        </html>
    """)


def export_computation_graph(
    closed_jaxpr: Any,
    format: str = "text",
) -> str:
    """Serialize a ClosedJaxpr to a human-readable or tool-consumable form.

    Args:
        closed_jaxpr: A :class:`jax.core.ClosedJaxpr` returned by
            :func:`jax.make_jaxpr`.
        format: Output format — one of ``"text"`` (default), ``"dot"``
            (Graphviz), or ``"html"`` (standalone HTML page).

    Returns:
        String representation in the requested format.

    Raises:
        ValueError: If *format* is not ``"text"``, ``"dot"``, or ``"html"``.

    Example::

        import jax, jax.numpy as jnp
        from stochax_market.debug import export_computation_graph

        closed = jax.make_jaxpr(lambda x: x ** 2)(jnp.ones(4))
        print(export_computation_graph(closed, format="dot"))
    """
    fmt = format.lower()
    if fmt == "text":
        buf = io.StringIO()
        buf.write("=== Computation Graph (text) ===\n")
        buf.write("Inputs:\n")
        for i, var in enumerate(closed_jaxpr.jaxpr.invars):
            buf.write(f"  [{i}] {var}  shape={var.aval}\n")
        buf.write("Equations:\n")
        for eq in closed_jaxpr.jaxpr.eqns:
            lhs = ", ".join(str(v) for v in eq.outvars)
            rhs = ", ".join(str(v) for v in eq.invars)
            buf.write(f"  {lhs} = {eq.primitive.name}({rhs})\n")
        buf.write("Outputs:\n")
        for i, var in enumerate(closed_jaxpr.jaxpr.outvars):
            aval = var.aval if hasattr(var, "aval") else "?"
            buf.write(f"  [{i}] {var}  shape={aval}\n")
        return buf.getvalue()

    if fmt == "dot":
        return _jaxpr_to_dot(closed_jaxpr)

    if fmt == "html":
        return _jaxpr_to_html(closed_jaxpr)

    raise ValueError(
        f"Unknown format {format!r}.  Choose one of: 'text', 'dot', 'html'."
    )
