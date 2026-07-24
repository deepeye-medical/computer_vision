"""Import images whose CSV labels contain path, label, and split."""

import csv
import os
from pathlib import Path

from PIL import Image

from computer_vision.config import parse_and_validate
from computer_vision.data_preparation.config import SourceConfig
from computer_vision.manifest import ImageRecord, write_manifest


def prepare(config: SourceConfig) -> None:
    """Check labelled images and preserve source-specific sample identity."""
    with (config.source / "labels.csv").open(newline="") as stream:
        labels = list(csv.DictReader(stream))
    records = []
    for row in labels:
        if "group_id" in row and not row["group_id"].strip():
            raise ValueError("Group metadata must identify every image")
        split = row["split"]
        if split not in {"train", "test"}:
            raise ValueError("Image split must be train or test")
        relative = Path(row["path"])
        image_path = (config.source / relative).resolve()
        if not image_path.is_relative_to(config.source.resolve()):
            raise ValueError("Image paths must stay inside the source directory")
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        records.append(
            ImageRecord(
                sample_id=f"{config.source_name}:{relative.as_posix()}",
                source=config.source_name,
                path=os.path.relpath(image_path, config.output.parent.resolve()),
                label=row["label"],
                split=split,
                width=width,
                height=height,
                group_id=row.get("group_id", ""),
            )
        )
    if not records:
        raise ValueError("Source labels must contain images")
    write_manifest(config.output, records)


def main() -> None:
    """Prepare a CSV-labelled image source."""
    prepare(parse_and_validate(SourceConfig))


if __name__ == "__main__":
    main()
