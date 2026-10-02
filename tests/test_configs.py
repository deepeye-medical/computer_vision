"""Check reusable configuration overlays and generated reference settings."""

import runpy
from pathlib import Path

import pytest

from computer_vision.train import TrainConfig, create_training_parser, instantiate_training_config


@pytest.mark.parametrize(
    "overlay",
    [
        "wandb",
        "model_resnet",
        "model_resnet_3d",
        "model_imagenet_resnet",
        "model_resnet34_image",
        "model_resnet18_slice",
        "model_resnet34_slice",
        "model_resnet18_volume",
        "model_resnet34_volume",
        "model_r2plus1d_18",
        "model_cnn",
        "model_cnn_slice",
        "model_cnn_volume",
        "model_vit",
        "model_imagenet_vit",
        "model_vit_slice",
        "model_vit_volume",
        "train_cosine",
        "train_elr",
        "loss_bce",
        "loss_focal",
        "loss_gce",
        "augmentation_mild",
        "train_manifest",
    ],
)
def test_overlay(overlay: str) -> None:
    """Compose each example with the base configuration and build its network."""
    parser = create_training_parser(TrainConfig)
    parsed = parser.parse_args(["--config", "configs/train.yaml", "--config", f"configs/{overlay}.yaml"])
    config, network = instantiate_training_config(parser, TrainConfig, parsed, initialize_pretrained=False)
    assert config.dataset.num_classes == network.num_classes


def test_generated_reference() -> None:
    """Keep documented defaults synchronized with the configuration schema."""
    generator = runpy.run_path("scripts/generate_config_examples.py")
    assert Path("configs/train.example.yaml").read_text() == generator["render_training_defaults"]()
