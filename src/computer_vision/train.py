"""Main entry point for image classification training."""

import json
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeVar, cast

import lightning.pytorch as pl
import yaml
from clearml import Task
from clearml.config import config as clearml_config
from coolname import generate_slug
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from jsonargparse.typing import register_type
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import Field

from computer_vision.config import ConfigModel
from computer_vision.data import DatasetConfig, ImageDataModule
from computer_vision.lightning_module import ClassificationModule, ConfigCheckpoint, TrainingConfig
from computer_vision.loss import BCEConfig, CrossEntropyConfig, FocalConfig, GCEConfig
from computer_vision.model import ResNet, VisionNetwork, ViT
from computer_vision.tracking import WandbConfig, wandb_run

# Replace loss variants completely when applying YAML overlays.
for loss_config_type in (BCEConfig, CrossEntropyConfig, FocalConfig, GCEConfig):
    register_type(loss_config_type, serializer=loss_config_type.model_dump, deserializer=loss_config_type.model_validate)


class TrainConfig(ConfigModel):
    """Top-level configuration for an image classification training run."""

    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    """Non-negative seed used by Python, NumPy, PyTorch, and data-loader workers."""

    resume_from: Path | None = None
    """Optional checkpoint from which to resume optimizer and training state."""

    output_dir: Path = Path("runs/training")
    """Directory receiving logs and checkpoints."""

    training: TrainingConfig = Field(default_factory=TrainingConfig)
    """Loss and optimizer settings."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    """Dataset and augmentation settings."""

    max_epochs: int = Field(default=100, ge=1)
    """Maximum number of complete training epochs."""

    early_stopping_patience: int = Field(default=10, ge=1)
    """Number of unimproved validation epochs tolerated before stopping."""

    accelerator: Literal["auto", "cpu", "gpu", "mps"] = "auto"
    """Lightning accelerator used for training."""

    devices: int = Field(default=1, ge=1)
    """Number of accelerator devices used by the trainer."""

    precision: Literal["32-true", "16-mixed", "bf16-mixed"] = "32-true"
    """Numerical precision policy used by Lightning."""

    clearml_project_name: str | None = None
    """ClearML project receiving the training task."""

    wandb: WandbConfig | None = None
    """Optional Weights & Biases settings; null disables tracking."""


TrainConfigT = TypeVar("TrainConfigT", bound=TrainConfig)


def create_training_parser(config_type: type[TrainConfigT]) -> ArgumentParser:
    """Create a parser for settings and a configurable image classification network."""
    parser = ArgumentParser(description=config_type.__doc__)
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_class_arguments(config_type)
    parser.add_subclass_arguments(
        VisionNetwork,
        "network",
        default="computer_vision.model.ResNet",
    )
    parser.link_arguments("dataset.num_classes", "network.init_args.num_classes", apply_on="instantiate")
    return parser


def instantiate_training_config(
    parser: ArgumentParser,
    config_type: type[TrainConfigT],
    parsed_values: Namespace,
    *,
    initialize_pretrained: bool = True,
) -> tuple[TrainConfigT, VisionNetwork]:
    """Instantiate the selected network and validate the remaining settings."""
    if (not initialize_pretrained or parsed_values.resume_from is not None) and hasattr(parsed_values.network.init_args, "pretrained"):
        parsed_values = deepcopy(parsed_values)
        parsed_values.network.init_args.pretrained = False
    instantiated_values = parser.instantiate(parsed_values).as_dict()
    instantiated_values.pop("config", None)
    network = cast(VisionNetwork, instantiated_values.pop("network"))
    config = config_type.model_validate(instantiated_values)
    if isinstance(network, (ResNet, ViT)) and initialize_pretrained and parsed_values.resume_from is None:
        network.load_pretrained_weights()
    image_size = getattr(network, "image_size", None)
    if image_size is not None and config.dataset.image_size != image_size:
        raise ValueError(f"Selected network requires dataset.image_size={image_size}")
    return config, network


def dump_training_config(
    parser: ArgumentParser,
    parsed_values: Namespace,
    config: TrainConfig,
) -> dict[str, object]:
    """Return serializable settings before class instantiation."""
    values = config.model_dump(mode="json", exclude_computed_fields=True)
    values["network"] = parsed_values.network
    return cast(dict[str, object], json.loads(parser.dump(parser.parse_object(values), format="json")))


def create_run_id() -> str:
    """Create a time-sortable identifier for a training run."""
    start_time = datetime.now().strftime("%Y%m%d%H%M%S")
    random_name = generate_slug(2).replace("-", "_")
    return f"training_{start_time}_{random_name}"


def main() -> None:
    """Configure and start an image classification training run."""
    parser = create_training_parser(TrainConfig)
    parsed_values = parser.parse_args()
    run_id = create_run_id()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    task = None
    if parsed_values.clearml_project_name is not None:
        Task.set_random_seed(None)
        # Use threads to avoid the SDK subprocess shutdown race.
        clearml_config.get("development")["report_use_subprocess"] = False
        task = Task.init(project_name=parsed_values.clearml_project_name, task_name=run_id, reuse_last_task_id=False)
    try:
        pl.seed_everything(parsed_values.seed, workers=True)
        config, network = instantiate_training_config(parser, TrainConfig, parsed_values)
        resolved_config = dump_training_config(parser, parsed_values, config)
        if task is not None:
            task.connect_configuration(resolved_config, name="training", ignore_remote_overrides=True)
        run_dir = config.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved_config))

        data_module = ImageDataModule(config.dataset)
        data_module.setup("fit")
        module = ClassificationModule(config.training, network=network)

        # ClearML automatically captures metrics and images written through TensorBoard.
        logger = TensorBoardLogger(save_dir=run_dir, name="tensorboard")
        with wandb_run(config.wandb, run_dir, resolved_config) as wandb_logger:
            checkpoint = ModelCheckpoint(
                dirpath=run_dir / "checkpoints",
                filename="best",
                monitor="accuracy/validation",
                mode="max",
                save_top_k=1,
                save_last=True,
            )
            early_stopping = EarlyStopping(
                monitor="accuracy/validation",
                mode="max",
                patience=config.early_stopping_patience,
            )
            trainer = pl.Trainer(
                max_epochs=config.max_epochs,
                accelerator=config.accelerator,
                devices=config.devices,
                precision=config.precision,
                default_root_dir=run_dir,
                logger=[logger, wandb_logger] if wandb_logger is not None else logger,
                callbacks=[
                    checkpoint,
                    early_stopping,
                    ConfigCheckpoint(resolved_config),
                ],
            )
            trainer.fit(module, datamodule=data_module, ckpt_path=config.resume_from)
    finally:
        if task is not None:
            task.close()


if __name__ == "__main__":
    main()
