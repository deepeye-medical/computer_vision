"""Generate two small synthetic image sources for the reference workflow."""

import csv
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import Field

from computer_vision.config import ConfigModel, parse_and_validate


class DummyConfig(ConfigModel):
    """Synthetic input settings."""

    output: Path = Path(".artifacts/dummy")
    samples_per_class: int = Field(default=6, ge=3)
    seed: int = Field(default=42, ge=0)


def generate(config: DummyConfig) -> None:
    """Create CSV-labelled and class-folder sources with repeated groups."""
    if config.output.exists():
        raise ValueError("Dummy output already exists; select a new directory")
    generator = np.random.default_rng(config.seed)
    rows = []
    groups = []
    for source in ("source_a", "source_b"):
        for label_index, label in enumerate(("circle", "square")):
            for index in range(config.samples_per_class):
                split = "test" if index == config.samples_per_class - 1 else "train"
                group_id = f"{source}:{label}:{split}:{index // 2}"
                relative = Path("images") / f"{label}-{index}.png" if source == "source_a" else Path(split) / label / f"{index}.png"
                image_path = config.output / source / relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                pixels = generator.integers(0, 40, (40, 48, 3), dtype=np.uint8)
                if label == "circle":
                    rows_grid, columns_grid = np.ogrid[:40, :48]
                    pixels[(rows_grid - 20) ** 2 + (columns_grid - 24) ** 2 < 12**2, label_index] = 220
                else:
                    pixels[8:32, 12:36, label_index] = 220
                Image.fromarray(pixels).save(image_path)
                if source == "source_a":
                    rows.append((relative.as_posix(), label, split, group_id))
                else:
                    groups.append((relative.as_posix(), group_id))
    with (config.output / "source_a/labels.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "label", "split", "group_id"))
        writer.writerows(rows)
    with (config.output / "source_b/groups.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "group_id"))
        writer.writerows(groups)
    logging.getLogger(__name__).info("Created synthetic sources in %s", config.output)


def main() -> None:
    """Generate the configured synthetic sources."""
    logging.basicConfig(level=logging.INFO)
    generate(parse_and_validate(DummyConfig))


if __name__ == "__main__":
    main()
