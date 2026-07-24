"""Common image records for preparation, training, and analysis."""

import csv
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from computer_vision.config import ConfigModel


class ImageRecord(ConfigModel):
    """Image record with a source-qualified sample ID."""

    sample_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    path: str = Field(min_length=1)
    label: str
    split: Literal["train", "test"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    group_id: str = ""

    @field_validator("group_id")
    @classmethod
    def normalize_group_id(cls, value: str) -> str:
        """Remove boundary whitespace so group identity stays consistent."""
        if value and not value.strip():
            raise ValueError("Group ID must not contain only whitespace")
        return value.strip()


def read_manifest(path: Path) -> list[ImageRecord]:
    """Read and validate records at the pipeline boundary."""
    with path.open(newline="") as stream:
        records = [ImageRecord.model_validate(row) for row in csv.DictReader(stream)]
    if not records:
        raise ValueError("Manifest must contain images")
    return records


def write_manifest(path: Path, records: list[ImageRecord]) -> None:
    """Write records with paths relative to the manifest directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ImageRecord.model_fields))
        writer.writeheader()
        writer.writerows(record.model_dump() for record in records)
