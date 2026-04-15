"""Pytree / Equinox module visualization utilities.

``display_module`` pretty-prints the tree structure of any :class:`eqx.Module`
(or generic JAX pytree), showing leaf shapes, dtypes, and — for Equinox modules
— whether each leaf is *trainable* (a JAX array) or *frozen* (a static Python
value).
"""

from __future__ import annotations

import io
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp


def _leaf_summary(leaf: Any) -> str:
    """Return a compact shape/dtype string for a pytree leaf."""
    if isinstance(leaf, jnp.ndarray):
        return f"Array{list(leaf.shape)} {leaf.dtype}"
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        return f"Array{list(leaf.shape)} {leaf.dtype}"
    return repr(leaf)


def _walk_module(
    module: Any,
    buf: io.StringIO,
    prefix: str = "",
    is_last: bool = True,
    *,
    _seen: set[int] | None = None,
) -> None:
    """Recursively render a module/pytree node into *buf*."""
    if _seen is None:
        _seen = set()

    connector = "└── " if is_last else "├── "
    child_prefix = prefix + ("    " if is_last else "│   ")

    # Equinox Module: walk named fields
    if isinstance(module, eqx.Module):
        obj_id = id(module)
        if obj_id in _seen:
            buf.write(f"{prefix}{connector}{type(module).__name__} (already shown)\n")
            return
        _seen.add(obj_id)

        buf.write(f"{prefix}{connector}{type(module).__name__}\n")

        # Separate leaves from sub-modules using equinox's partition
        dynamic, static = eqx.partition(module, eqx.is_array)

        fields = list(vars(module).items())
        for i, (name, value) in enumerate(fields):
            last = i == len(fields) - 1
            child_conn = "└── " if last else "├── "
            child_child_prefix = child_prefix + ("    " if last else "│   ")

            if isinstance(value, eqx.Module):
                buf.write(f"{child_prefix}{child_conn}{name}:\n")
                _walk_module(value, buf, child_prefix, last, _seen=_seen)
            elif eqx.is_array(value):
                buf.write(
                    f"{child_prefix}{child_conn}"
                    f"{name}: {_leaf_summary(value)}  [trainable]\n"
                )
            else:
                buf.write(
                    f"{child_prefix}{child_conn}"
                    f"{name}: {_leaf_summary(value)}  [static]\n"
                )
    else:
        # Plain pytree (list, tuple, dict, leaf)
        leaves, treedef = jax.tree_util.tree_flatten(module)
        if leaves:
            buf.write(f"{prefix}{connector}{treedef!r}\n")
            for i, leaf in enumerate(leaves):
                last = i == len(leaves) - 1
                leaf_conn = "└── " if last else "├── "
                buf.write(
                    f"{child_prefix}{leaf_conn}leaf: {_leaf_summary(leaf)}\n"
                )
        else:
            buf.write(f"{prefix}{connector}{module!r}\n")


def display_module(module: Any) -> str:
    """Pretty-print an Equinox module (or pytree) tree with shapes and roles.

    Walks the pytree/module structure recursively, displaying:

    * Class names for :class:`eqx.Module` nodes.
    * Leaf shapes and dtypes for JAX arrays (annotated ``[trainable]``).
    * Static Python values (annotated ``[static]``).

    Args:
        module: An :class:`eqx.Module` instance or any JAX-registered pytree.

    Returns:
        Multi-line string representation of the module tree.

    Example::

        from stochax_market.model.spde import SPDEStepper
        from stochax_market.debug import display_module
        print(display_module(SPDEStepper()))
    """
    buf = io.StringIO()
    root_name = type(module).__name__ if hasattr(module, "__class__") else "root"
    buf.write(f"{root_name}\n")
    _walk_module(module, buf, prefix="", is_last=True)
    return buf.getvalue()
