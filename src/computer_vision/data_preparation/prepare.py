"""Prepare a class-folder source as a training manifest."""

from pathlib import Path

from computer_vision.config import parse_and_validate
from computer_vision.data_preparation.source_b.prepare import FolderSourceConfig
from computer_vision.data_preparation.source_b.prepare import prepare as prepare_folder


class PreparationConfig(FolderSourceConfig):
    """Paths and source identity for the class-folder preparation command."""

    output: Path = Path(".artifacts/images.csv")


def prepare(config: PreparationConfig) -> None:
    """Write a training manifest without changing source images."""
    prepare_folder(config)


def main() -> None:
    """Prepare the configured class-folder source."""
    prepare(parse_and_validate(PreparationConfig))


if __name__ == "__main__":
    main()
