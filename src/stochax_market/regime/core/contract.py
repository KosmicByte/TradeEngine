"""Point-in-time data contract — the single sanctioned input to detection.

A :class:`PointInTimeFrame` guarantees *by construction* that no information dated after
``as_of`` is visible to the model. Look-ahead bias — letting a future bar influence a past
decision — is the single most common way a regime backtest silently lies, so we make it
structurally impossible rather than a thing to remember.

The frame truncates to ``as_of`` at construction time: once built, there is no API that
returns a bar later than ``as_of``. Detection, transition scoring, and walk-forward
evaluation all consume frames, so the guarantee holds everywhere uniformly.

Exogenous (macro / cross-asset) features added in a later phase carry their own
``known_at`` lag — a value becomes visible only once it would actually have been published
and would not be revised. The hook for that lives in :meth:`PointInTimeFrame.with_exogenous`,
which is intentionally a stub today: building the door now is cheap, walking through it waits
until the point-in-time machinery is in place and proven.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PointInTimeFrame:
    """An immutable, truncated-to-``as_of`` view of a price series.

    Construct via :meth:`from_dataframe` (preferred — consumes ``data.load`` output) or
    :meth:`from_arrays`. Direct construction is supported but the classmethods do the
    defensive coercion (sorting, NaN handling, truncation) you almost always want.

    Attributes:
        close: 1-D float array of close prices, ascending in time, already truncated so the
            last element is at or before ``as_of``.
        timestamps: matching ``DatetimeIndex`` (same length as ``close``).
        as_of: the evaluation instant. Nothing dated after this is visible. This is the
            timestamp that flows through to ``RegimeResult.as_of``.
        symbol, interval: metadata carried through to the result.
    """

    close: np.ndarray
    timestamps: pd.DatetimeIndex
    as_of: datetime
    symbol: str = "?"
    interval: str = "1d"

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        as_of: datetime | None = None,
        close_col: str = "close",
        ts_col: str = "ts",
        symbol: str | None = None,
        interval: str | None = None,
    ) -> PointInTimeFrame:
        """Build from a ``data.load``-style frame (columns: symbol, interval, ts, close, ...).

        Args:
            df: source frame. Must contain ``close_col`` and ``ts_col``.
            as_of: evaluation instant. Defaults to the latest timestamp in ``df`` — i.e. a
                nowcast on the most recent available bar.
            close_col, ts_col: column names to read.
            symbol, interval: override metadata; otherwise read from the frame if present.
        """
        if close_col not in df.columns or ts_col not in df.columns:
            raise ValueError(
                f"PointInTimeFrame.from_dataframe needs columns '{close_col}' and '{ts_col}'; "
                f"got {list(df.columns)}."
            )
        ts = pd.to_datetime(df[ts_col].to_numpy())
        close = np.asarray(df[close_col].to_numpy(), dtype=float)
        sym = symbol
        if sym is None and "symbol" in df.columns and len(df) > 0:
            sym = str(df["symbol"].iloc[0])
        if interval is None and "interval" in df.columns and len(df) > 0:
            interval = str(df["interval"].iloc[0])
        return cls.from_arrays(
            close=close,
            timestamps=ts,
            as_of=as_of,
            symbol=sym or "?",
            interval=interval or "1d",
        )

    @classmethod
    def from_arrays(
        cls,
        close: np.ndarray,
        timestamps: pd.Series | pd.DatetimeIndex | np.ndarray | None = None,
        *,
        as_of: datetime | None = None,
        symbol: str = "?",
        interval: str = "1d",
    ) -> PointInTimeFrame:
        """Build from a raw close array and optional timestamps.

        When ``timestamps`` is omitted we synthesise a daily index ending today; this keeps
        the array-only call sites (and synthetic series) working while still routing them
        through the same truncation guarantee.
        """
        close = np.asarray(close, dtype=float)
        if close.ndim != 1:
            raise ValueError(f"close must be 1-D; got shape {close.shape}.")
        if timestamps is None:
            # Synthetic ascending daily index ending today. The values are arbitrary but
            # monotonic, so truncation logic and as_of remain well-defined for raw arrays.
            end = pd.Timestamp(datetime.now().date())
            idx = pd.date_range(end=end, periods=len(close), freq="D")
        else:
            idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(timestamps)))
        if len(idx) != len(close):
            raise ValueError(
                f"close and timestamps length mismatch: {len(close)} vs {len(idx)}."
            )
        # Sort ascending defensively.
        order = np.argsort(idx.values)
        idx = idx[order]
        close = close[order]
        # Resolve as_of and TRUNCATE — this is the guarantee.
        resolved_as_of = (
            pd.Timestamp(as_of) if as_of is not None else (idx[-1] if len(idx) else pd.Timestamp(datetime.now()))
        )
        mask = idx <= resolved_as_of
        idx = idx[mask]
        close = close[mask]
        if len(close) == 0:
            raise ValueError(
                f"No bars at or before as_of={resolved_as_of}. The frame would be empty — "
                f"check that as_of is not earlier than the first bar."
            )
        return cls(
            close=close,
            timestamps=idx,
            as_of=resolved_as_of.to_pydatetime(),
            symbol=symbol,
            interval=interval,
        )

    # ------------------------------------------------------------------ views
    @property
    def n(self) -> int:
        """Number of visible bars."""
        return int(len(self.close))

    @property
    def last_ts(self) -> datetime:
        """Timestamp of the most recent visible bar (always <= as_of)."""
        return self.timestamps[-1].to_pydatetime()

    def tail(self, n: int) -> PointInTimeFrame:
        """A point-in-time-safe view of the most recent ``n`` bars.

        Still truncated to the same ``as_of`` — taking a tail cannot reveal future data.
        Detection windows should be carved with this rather than by slicing raw arrays.
        """
        if n <= 0:
            raise ValueError("tail(n) requires n > 0.")
        n = min(n, self.n)
        return PointInTimeFrame(
            close=self.close[-n:],
            timestamps=self.timestamps[-n:],
            as_of=self.as_of,
            symbol=self.symbol,
            interval=self.interval,
        )

    # ------------------------------------------------------------------ provenance
    def input_hash(self) -> str:
        """Stable SHA-256 over the visible inputs.

        Deterministic across processes and machines for identical data, so two runs that
        produce the same hash provably saw the same inputs. Covers close values, timestamps,
        as_of, symbol, and interval — everything that can change the output.
        """
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.close, dtype="<f8").tobytes())
        h.update(self.timestamps.asi8.astype("<i8").tobytes())
        h.update(str(self.as_of).encode())
        h.update(self.symbol.encode())
        h.update(self.interval.encode())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------------ future hook
    def with_exogenous(self, *_args: object, **_kwargs: object) -> PointInTimeFrame:
        """Reserved for macro / cross-asset features with ``known_at`` lag semantics.

        Intentionally not implemented yet. When exogenous features land (post-R2), each
        feature column will carry the timestamp at which it would actually have been
        published; this method will attach them and the truncation logic will additionally
        mask any feature value whose ``known_at`` is after ``as_of``. Documented now so the
        contract's shape is stable for downstream consumers.
        """
        raise NotImplementedError(
            "Exogenous (macro/cross-asset) features are planned for a later phase. "
            "The point-in-time contract reserves this hook so they can be added without a "
            "breaking change."
        )
