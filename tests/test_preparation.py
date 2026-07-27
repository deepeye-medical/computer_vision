"""Check the portable preparation artifact."""

import csv
import os
from pathlib import Path

from PIL import Image

from computer_vision.data_preparation.prepare import PreparationConfig, prepare


def test_manifest(tmp_path: Path) -> None:
    """Write relative image paths without changing the source."""
    source = tmp_path / "source"
    image_path = source / "train" / "sample_class" / "image.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (20, 16)).save(image_path)
    original = image_path.read_bytes()
    output = tmp_path / ".artifacts" / "images.csv"
    prepare(PreparationConfig(source=source, output=output))
    with output.open() as stream:
        records = list(csv.DictReader(stream))
    assert records == [
        {
            "sample_id": "source_b:train/sample_class/image.png",
            "source": "source_b",
            "path": os.path.relpath(image_path, output.parent),
            "split": "train",
            "label": "sample_class",
            "width": "20",
            "height": "16",
            "group_id": "",
        }
    ]
    assert image_path.read_bytes() == original
