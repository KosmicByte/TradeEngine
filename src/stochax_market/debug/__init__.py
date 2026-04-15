"""stochax_market.debug — unified debugging & inspection utilities.

Public API
----------

.. autofunction:: stochax_market.debug.display_module
.. autofunction:: stochax_market.debug.display_jaxpr
.. autofunction:: stochax_market.debug.display_math_eq
.. autofunction:: stochax_market.debug.inject_debug_prints
.. autofunction:: stochax_market.debug.export_computation_graph
"""

from __future__ import annotations

from stochax_market.debug.inject import inject_debug_prints
from stochax_market.debug.jaxpr import display_jaxpr, export_computation_graph
from stochax_market.debug.math_eq import display_math_eq
from stochax_market.debug.module import display_module

__all__ = [
    "display_module",
    "display_jaxpr",
    "display_math_eq",
    "inject_debug_prints",
    "export_computation_graph",
]
