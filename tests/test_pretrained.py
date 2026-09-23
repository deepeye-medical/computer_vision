"""Check torchvision adapters without downloading pretrained weights."""

from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime
import pytest
import torch
from torch import nn
from torchvision import models

from computer_vision.artifacts import export_network_to_onnx, export_network_to_torch_export
from computer_vision.evaluate import load_checkpoint
from computer_vision.lightning_module import ClassificationModule
from computer_vision.model import TorchvisionClassifier
from computer_vision.train import TrainConfig, create_training_parser, dump_training_config, instantiate_training_config


@pytest.fixture(params=["resnet18", "vit_b_16"])
def backbone(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Literal["resnet18", "vit_b_16"]:
    """Use ResNet and a small torchvision transformer without downloads."""
    original_builder = models.get_model

    def build_model(name: str, *, weights: object = None) -> nn.Module:
        assert weights is None, "Tests must not download pretrained weights"
        if name == "vit_b_16":
            return models.VisionTransformer(image_size=224, patch_size=16, num_layers=1, num_heads=2, hidden_dim=32, mlp_dim=64)
        return original_builder(name, weights=None)

    monkeypatch.setattr(models, "get_model", build_model)
    torch.set_num_threads(1)
    return request.param


def test_frozen_backbone_checkpoint(backbone: Literal["resnet18", "vit_b_16"], tmp_path: Path) -> None:
    """Train the task head, preserve normalization, and restore without downloads."""
    parser = create_training_parser(TrainConfig)
    parsed = parser.parse_args(
        [
            "--config",
            "configs/synthetic.yaml",
            "--dataset.image_size",
            "224",
            "--network.class_path",
            "computer_vision.model.TorchvisionClassifier",
            "--network.init_args.backbone",
            backbone,
            "--network.init_args.freeze_backbone",
            "true",
        ]
    )
    config, network = instantiate_training_config(parser, TrainConfig, parsed, initialize_pretrained=False)
    assert isinstance(network, TorchvisionClassifier)
    network.train()
    assert not network.backbone.training
    images = torch.rand(2, 3, 224, 224)
    logits = network(images)
    logits.sum().backward()
    assert logits.shape == (2, 2)
    assert sum(parameter.requires_grad for parameter in network.parameters()) == 2
    assert all(parameter.grad is not None for parameter in network.parameters() if parameter.requires_grad)
    checkpoint = tmp_path / "model.ckpt"
    module = ClassificationModule(config.training, network).eval()
    torch.save(
        {
            "run_config": dump_training_config(parser, parsed, config),
            "state_dict": module.state_dict(),
            "class_names": ["negative", "positive"],
        },
        checkpoint,
    )
    restored, restored_config = load_checkpoint(checkpoint)
    restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(images), module(images))
        expected = network.backbone((images - network.input_mean) / network.input_std)
        torch.testing.assert_close(network(images), expected)
    assert restored_config == config
    assert restored.class_names == ("negative", "positive")


def test_backbone_exports(backbone: Literal["resnet18", "vit_b_16"], tmp_path: Path) -> None:
    """Preserve normalized logits in both export formats with dynamic batches."""
    network = TorchvisionClassifier(backbone=backbone, pretrained=False).eval()
    example = torch.rand(2, 3, 224, 224)
    exported_path = tmp_path / "network.pt2"
    export_network_to_torch_export(network, example, exported_path)
    onnx_path = tmp_path / "network.onnx"
    export_network_to_onnx(network, example, onnx_path, input_name="images")
    exported = torch.export.load(exported_path).module()
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    images = torch.rand(1, 3, 224, 224)
    with torch.inference_mode():
        expected = network(images)
        torch.testing.assert_close(exported(images), expected)
    actual = np.asarray(session.run(None, {"images": images.numpy()})[0])
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-4, atol=1e-5)


def test_imagenet_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load backbone weights and replace their head without a network request."""
    state = models.resnet18(weights=None).state_dict()
    state["conv1.weight"].fill_(0.25)

    def load_weights(*_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        return state

    monkeypatch.setattr(models.ResNet18_Weights, "get_state_dict", load_weights)
    network = TorchvisionClassifier(pretrained=True, num_classes=3)
    assert isinstance(network.backbone, models.ResNet)
    torch.testing.assert_close(network.backbone.conv1.weight, state["conv1.weight"])
    assert network.backbone.fc.out_features == 3
