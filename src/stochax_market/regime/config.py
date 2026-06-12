"""Regime Radar configuration (ported from standalone RegimeRadar).

MERGE NOTE (Ares Alpha): this is a standalone Settings object preserving the original
REGIME_* env namespace so the ported module runs self-contained. At full integration into
stochax_market, these knobs should fold into the unified Ares config under a `regime`
section. Until then this keeps detection runnable with the original .env contract.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Override via environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="REGIME_",
        extra="ignore",
    )

    # Data
    marketlake_data_dir: Path | None = Field(
        default=None,
        description="If set and importable, MarketLake is used in preference to yfinance.",
    )
    default_symbol: str = Field(default="^NSEI", description="Yahoo symbol for NIFTY50.")
    default_interval: str = Field(default="1d", description="Bar interval (1d / 1h / 5m).")

    # Analysis
    window: int = Field(default=126, description="Rolling window length in bars (~6 months daily).")
    edmd_rank: int = Field(default=10, description="Truncation rank for SVD in (E)DMD.")
    hmm_n_states: int = Field(default=3, description="Number of HMM regimes.")

    # Transition risk
    transition_lookback: int = Field(default=20, description="Bars over which mode drift is measured.")
    transition_threshold: float = Field(
        default=0.6,
        description="Risk score above which a regime transition is flagged.",
    )

    # Output
    artifacts_dir: Path = Field(default=Path("./artifacts"))

    # Provenance / reproducibility (R0)
    provenance_enabled: bool = Field(
        default=False,
        description="If true, detect_regime appends an immutable InferenceRecord per call.",
    )
    provenance_dir: Path = Field(
        default=Path("./artifacts/provenance"),
        description="Directory for append-only JSONL inference logs.",
    )

    # Calibration & uncertainty (R1)
    calibration_enabled: bool = Field(
        default=True,
        description="If true, detection uses the calibration artifact when one is present.",
    )
    calibration_artifact: Path = Field(
        default=Path("./artifacts/calibration.json"),
        description="Path to the fitted calibration artifact (created by `regime calibrate`).",
    )
    raw_hmm_dampening: bool = Field(
        default=True,
        description=(
            "Keep the hand-tuned HMM-confidence dampening ladder. Default True preserves R0 "
            "behaviour; R2 will A/B this against calibration to retire it."
        ),
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Cached settings accessor."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return _settings
