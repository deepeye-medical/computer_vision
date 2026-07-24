"""Combine source manifests into one training table."""

import logging
import os
from pathlib import Path

from computer_vision.config import ConfigModel, parse_and_validate
from computer_vision.manifest import read_manifest, write_manifest


class IntegrationConfig(ConfigModel):
    """Input manifests and cached output location."""

    manifests: tuple[Path, ...]
    output: Path = Path(".artifacts/images.csv")


def integrate(config: IntegrationConfig) -> None:
    """Rebase paths and reject repeated identities or train/test image overlap."""
    records = []
    identities = set()
    image_paths = set()
    split_by_group = {}
    for manifest in config.manifests:
        for record in read_manifest(manifest):
            image_path = (manifest.parent / record.path).resolve()
            if record.sample_id in identities or image_path in image_paths:
                raise ValueError("Integration requires unique sample IDs and image paths")
            identities.add(record.sample_id)
            image_paths.add(image_path)
            group_id = record.group_id or record.sample_id
            if group_id in split_by_group and split_by_group[group_id] != record.split:
                raise ValueError("Integration requires disjoint train and test groups")
            split_by_group[group_id] = record.split
            records.append(record.model_copy(update={"path": os.path.relpath(image_path, config.output.parent.resolve())}))
    if not records:
        raise ValueError("Integration requires at least one source manifest")
    write_manifest(config.output, records)
    logging.getLogger(__name__).info("Integrated %d images", len(records))


def main() -> None:
    """Integrate the configured sources."""
    logging.basicConfig(level=logging.INFO)
    integrate(parse_and_validate(IntegrationConfig))


if __name__ == "__main__":
    main()
