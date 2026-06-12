"""Immutable inference records for reproducibility.

Every detection can emit one append-only JSONL line capturing the input hash, the versions
in play, and the result. Given the record plus the data, a past decision is reproducible and
auditable — "why did the regime flip on 2025-03-14?" becomes a lookup, not a guess.

Records are deliberately small and dependency-free (stdlib ``json`` only) so the provenance
log stays cheap to write and trivial to read with any tool. Writing is opt-in via
``settings.provenance_enabled`` so library/eval use that runs thousands of detections does
not spew records unless asked.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class InferenceRecord:
    """One auditable detection event.

    Attributes:
        recorded_at_utc: wall-clock instant the record was written (UTC, ISO-8601).
        as_of: the point-in-time instant of the inputs (may be far in the past for a replay).
        symbol, interval: what was analysed.
        model_version: detection-logic version (see ``version.MODEL_VERSION``).
        code_version: running-code identifier (git SHA or package version).
        input_hash: stable hash of the point-in-time inputs.
        label: the regime label assigned.
        confidence: the label's probability (calibrated once R1 lands; raw until then).
        probabilities: full distribution over labels.
        extra: open slot for fields added by later phases (ood_score, prediction_set, ...).
    """

    as_of: str
    symbol: str
    interval: str
    model_version: str
    code_version: str
    input_hash: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    recorded_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    extra: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        """Compact single-line JSON suitable for a JSONL log."""
        return json.dumps(asdict(self), separators=(",", ":"), default=str)


def write_record(record: InferenceRecord, path: Path) -> None:
    """Append ``record`` as one line to the JSONL log at ``path`` (creating dirs as needed).

    Append-only and best-effort: provenance must never break a detection, so callers should
    treat a write failure as non-fatal. Concurrent appends of single short lines are safe on
    POSIX for the small payloads we write here.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(record.to_json() + "\n")


def read_records(path: Path) -> list[InferenceRecord]:
    """Read a JSONL provenance log back into records (for audit / replay / tests)."""
    path = Path(path)
    if not path.exists():
        return []
    out: list[InferenceRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(InferenceRecord(**json.loads(line)))
    return out
