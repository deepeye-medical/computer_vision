"""Summarize source coverage and image problems before training."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from computer_vision.config import ConfigModel, parse_and_validate
from computer_vision.manifest import read_manifest


class ReportConfig(ConfigModel):
    """Manifest and report paths."""

    manifest: Path
    output: Path = Path(".artifacts/report.json")


def report(config: ReportConfig) -> None:
    """Count classes, dimensions, missing labels, and duplicate image content."""
    records = read_manifest(config.manifest)
    content_counts: Counter[str] = Counter()
    content_splits: dict[str, set[str]] = {}
    missing_images = 0
    for record in records:
        image_path = config.manifest.parent / record.path
        if not image_path.is_file():
            missing_images += 1
            continue
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            content_counts[digest] += 1
            content_splits.setdefault(digest, set()).add(record.split)
    summary = {
        "samples": len(records),
        "classes": dict(sorted(Counter(record.label for record in records).items())),
        "sources": dict(sorted(Counter(record.source for record in records).items())),
        "splits": dict(sorted(Counter(record.split for record in records).items())),
        "image_sizes": dict(sorted(Counter(f"{record.width}x{record.height}" for record in records).items())),
        "missing_labels": sum(not record.label.strip() for record in records),
        "missing_images": missing_images,
        "duplicate_sample_ids": sum(count - 1 for count in Counter(record.sample_id for record in records).values()),
        "duplicate_images": sum(count - 1 for count in content_counts.values()),
        "train_test_duplicate_images": sum(len(splits) > 1 for splits in content_splits.values()),
        "groups_by_split": {
            split: len({record.group_id or record.sample_id for record in records if record.split == split}) for split in ("train", "test")
        },
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    """Write the configured analysis report."""
    report(parse_and_validate(ReportConfig))


if __name__ == "__main__":
    main()
