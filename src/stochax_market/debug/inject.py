"""Intermediate-value debug printing for JIT-compiled functions.

``inject_debug_prints`` wraps a jitted function and injects
:func:`jax.debug.print` calls that log named intermediate values at runtime
without breaking JIT compilation.

The debug keys refer to *positional argument indices* (``"arg0"``, ``"arg1"``,
…) or to named attributes of the first argument when it is an
:class:`eqx.Module` (e.g. ``"raw_mu_scale"``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax


def inject_debug_prints(
    jitted_fn: Callable,
    debug_keys: list[str],
) -> Callable:
    """Wrap *jitted_fn* to log intermediate values via :func:`jax.debug.print`.

    The wrapper intercepts every call, prints the requested values using
    :func:`jax.debug.print` (which is safe inside JIT), then delegates to the
    original function unchanged.

    *debug_keys* controls what is printed:

    * ``"arg0"``, ``"arg1"``, … — logs the whole positional argument (shape /
      value summary printed by JAX).
    * ``"arg0.field"`` — logs ``args[0].field`` when the argument is an
      :class:`eqx.Module` or plain object with that attribute.

    The wrapper is *pure* in the JAX sense: it does not change the return value
    or any external state beyond printing to stdout via
    :func:`jax.debug.print`.

    Args:
        jitted_fn: Any callable (jitted or not).  If not already jitted, the
            wrapper does not add JIT — it is the caller's responsibility.
        debug_keys: List of keys to inspect.  Examples:
            ``["arg0", "arg1.raw_mu_scale"]``.

    Returns:
        A new callable with the same signature as *jitted_fn* but with
        :func:`jax.debug.print` side-effects for each requested key.

    Example::

        import jax, jax.numpy as jnp
        from stochax_market.debug import inject_debug_prints

        @jax.jit
        def loss(x, y):
            return jnp.mean((x - y) ** 2)

        debug_loss = inject_debug_prints(loss, ["arg0", "arg1"])
        debug_loss(jnp.ones(4), jnp.zeros(4))
        # Prints: [debug] arg0: [1. 1. 1. 1.]
        #         [debug] arg1: [0. 0. 0. 0.]
    """
    def _get_value(args: tuple[Any, ...], key: str) -> Any | None:
        """Resolve *key* against *args*; return None if not found."""
        if "." in key:
            base, attr = key.split(".", 1)
        else:
            base, attr = key, None

        # Resolve base: "arg0" → args[0], "arg1" → args[1], etc.
        if base.startswith("arg") and base[3:].isdigit():
            idx = int(base[3:])
            if idx >= len(args):
                return None
            obj = args[idx]
        else:
            return None

        if attr is None:
            return obj

        return getattr(obj, attr, None)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        for key in debug_keys:
            val = _get_value(args, key)
            if val is None:
                jax.debug.print("[debug] {key}: <not found>", key=key)
            else:
                jax.debug.print("[debug] {key}: {val}", key=key, val=val)
        return jitted_fn(*args, **kwargs)

    # Preserve the original function's name/docs for introspection
    wrapper.__name__ = getattr(jitted_fn, "__name__", "wrapped")
    wrapper.__doc__ = getattr(jitted_fn, "__doc__", None)
    return wrapper
