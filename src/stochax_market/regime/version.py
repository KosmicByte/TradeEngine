"""Single source of truth for the detection-logic version and the running code version.

`MODEL_VERSION` is a semantic version for the *detection logic* — bump it whenever a change
could alter the label produced for an identical input. It is stamped onto every
``RegimeResult`` so a past call can be reproduced exactly and so champion/challenger
comparisons are unambiguous. It is deliberately separate from the package version in
``pyproject.toml`` / ``__init__`` (which tracks the distribution, not the decision logic).
"""

from __future__ import annotations

import subprocess
from functools import lru_cache

# Detection-logic version. Bump on ANY change that can alter outputs for identical input
# (new voter, changed thresholds, new calibration). Keep in lockstep with a CHANGELOG entry.
MODEL_VERSION = "1.1.1"


@lru_cache(maxsize=1)
def code_version() -> str:
    """Best-effort identifier of the running code.

    Prefers the short git SHA (with a ``+dirty`` marker when the working tree has
    uncommitted changes), so a provenance record points at exact source. Falls back to the
    installed package version, then to ``"unknown"``. Cached — resolved once per process.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout.strip()
        if sha:
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            ).stdout.strip()
            return f"git:{sha}{'+dirty' if dirty else ''}"
    except Exception:  # noqa: BLE001 — git absent / not a repo / timeout: fall through
        pass
    try:
        from stochax_market import __version__

        return f"pkg:{__version__}"
    except Exception:  # noqa: BLE001 — defensive
        return "unknown"
