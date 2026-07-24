"""Import images from source/{train,test}/class/image."""

import csv
import os
from pathlib import Path

from PIL import Image

from computer_vision.config import parse_and_validate
from computer_vision.data_preparation.config import SourceConfig
from computer_vision.manifest import ImageRecord, write_manifest


class FolderSourceConfig(SourceConfig):
    """Defaults for the class-folder source."""

    source_name: str = "source_b"
    output: Path = Path(".artifacts/source_b.csv")


def prepare(config: SourceConfig) -> None:
    """Record class-folder images without changing source files."""
    records = []
    groups_path = config.source / "groups.csv"
    groups = {}
    if groups_path.is_file():
        with groups_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                relative = Path(row["path"]).as_posix()
                if relative in groups:
                    raise ValueError("Group metadata must have unique image paths")
                groups[relative] = row["group_id"]
    for image_path in sorted(config.source.glob("*/*/*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        relative = image_path.relative_to(config.source)
        if groups_path.is_file() and not groups.get(relative.as_posix(), "").strip():
            raise ValueError("Group metadata must identify every image")
        split, label, _ = relative.parts
        if split not in {"train", "test"}:
            raise ValueError("Image split must be train or test")
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        records.append(
            ImageRecord(
                sample_id=f"{config.source_name}:{relative.as_posix()}",
                source=config.source_name,
                path=os.path.relpath(image_path.resolve(), config.output.parent.resolve()),
                label=label,
                split=split,
                width=width,
                height=height,
                group_id=groups.get(relative.as_posix(), ""),
            )
        )
    if not records:
        raise ValueError("Expected source/{train,test}/class/image files")
    write_manifest(config.output, records)


def main() -> None:
    """Prepare a class-folder image source."""
    prepare(parse_and_validate(FolderSourceConfig))


if __name__ == "__main__":
    main()
