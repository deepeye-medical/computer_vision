"""Evaluate a checkpoint using its saved model and preprocessing settings."""

from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import torch
from pydantic import Field

from computer_vision.config import ConfigModel, parse_and_validate
from computer_vision.data import ImageDataModule
from computer_vision.lightning_module import ClassificationModule
from computer_vision.train import TrainConfig, create_training_parser, instantiate_training_config


class EvaluationConfig(ConfigModel):
    """Checkpoint and optional runtime overrides."""

    checkpoint: Path
    root: Path | None = None
    accelerator: Literal["auto", "cpu", "gpu", "mps"] | None = None
    devices: int | None = Field(default=None, ge=1)
    precision: Literal["32-true", "16-mixed", "bf16-mixed"] | None = None
    output_dir: Path | None = None


def load_checkpoint(path: Path) -> tuple[ClassificationModule, TrainConfig]:
    """Rebuild the saved network and restore its weights."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    parser = create_training_parser(TrainConfig)
    config, network = instantiate_training_config(parser, TrainConfig, parser.parse_object(checkpoint["run_config"]), initialize_pretrained=False)
    module = ClassificationModule(config.training, network)
    module.load_state_dict(checkpoint["state_dict"])
    class_names = checkpoint.get("class_names")
    if class_names is not None:
        if (
            not isinstance(class_names, list)
            or not all(isinstance(name, str) for name in class_names)
            or len(class_names) != config.dataset.num_classes
        ):
            raise ValueError("Checkpoint class names must match its class count")
        module.class_names = tuple(class_names)
    return module, config


def main() -> None:
    """Evaluate held-out test images."""
    arguments = parse_and_validate(EvaluationConfig)
    module, config = load_checkpoint(arguments.checkpoint)
    dataset_config = config.dataset
    if arguments.root is not None:
        dataset_config = dataset_config.model_copy(update={"root": arguments.root})
    pl.seed_everything(config.seed, workers=True)
    trainer = pl.Trainer(
        accelerator=arguments.accelerator or config.accelerator,
        devices=arguments.devices or config.devices,
        precision=arguments.precision or config.precision,
        default_root_dir=arguments.output_dir or config.output_dir,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.test(module, datamodule=ImageDataModule(dataset_config))


if __name__ == "__main__":
    main()
