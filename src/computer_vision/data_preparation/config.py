"""Shared paths and identity settings for source adapters."""

from pathlib import Path

from computer_vision.config import ConfigModel


class SourceConfig(ConfigModel):
    """Source paths supplied through the CLI or DVC variables."""

    source: Path
    output: Path = Path(".artifacts/source_a.csv")
    source_name: str = "source_a"
