"""Check attribution math, model layouts, and saved explanation artifacts."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from computer_vision.interpretability import AttributionConfig, AttributionResult, attribute, cli
from computer_vision.interpretability.attribution import Method
from computer_vision.interpretability.visualization import save_visualizations
from computer_vision.lightning_module import ClassificationModule
from computer_vision.model.cnn import CNN, CNNStageConfig
from computer_vision.model.pretrained import TorchvisionClassifier
from computer_vision.model.resnet import ResNet, ResStageConfig, StemConfig
from computer_vision.model.vit import EncoderConfig, PatchConfig, ViT
from computer_vision.train import TrainConfig, create_training_parser, dump_training_config, instantiate_training_config


class LinearImageClassifier(nn.Module):
    """Return opposite linear class scores with known pixel contributions."""

    def __init__(self) -> None:
        """Build a positive pixel projection and an in-place activation."""
        super().__init__()
        self.features = nn.Conv2d(1, 1, 1, bias=False)
        nn.init.ones_(self.features.weight)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return positive and negative sums of stored intensities."""
        score = self.relu(self.features(images)).flatten(1).sum(dim=1)
        return torch.stack((score, -score), dim=1)


@pytest.mark.parametrize("method", ["gradcam", "saliency", "integrated_gradients", "occlusion"])
def test_exact_attribution(method: Method) -> None:
    """Recover analytic contributions and retain a requested negative class."""
    network = LinearImageClassifier()
    images = torch.linspace(0.1, 0.9, 30).reshape(2, 1, 3, 5)
    config = AttributionConfig(method=method, target_class=0, target_layer="features", baseline=0.05, steps=4, occlusion_size=1)
    result = attribute(network, images, config)
    expected = images if method == "gradcam" else torch.ones_like(images) if method == "saliency" else images - 0.05
    torch.testing.assert_close(result.maps, expected)
    assert result.targets.tolist() == [0, 0]
    if method == "integrated_gradients":
        torch.testing.assert_close(result.completeness_error, torch.zeros(2), atol=2e-6, rtol=0)
    if method in ("occlusion", "integrated_gradients"):
        negative = attribute(network, images, config.model_copy(update={"target_class": 1}))
        torch.testing.assert_close(negative.maps, -expected)


@pytest.mark.parametrize("architecture", ["cnn", "resnet", "vit"])
@pytest.mark.parametrize("mode", ["image", "slice", "volume"])
def test_gradcam_layouts(architecture: str, mode: str) -> None:
    """Map spatial features and ViT patches back to every input layout."""
    torch.manual_seed(42)
    torch.set_num_threads(1)
    dimensions = 3 if mode == "volume" else 2
    if architecture == "cnn":
        network = CNN(
            in_channels=1,
            slice_wise=mode == "slice",
            use_coordinate_channels=True,
            stages=(
                CNNStageConfig(out_channels=4, kernel_size=(3,) * dimensions, stride=(1,) * dimensions, padding=(0,) * dimensions, pool_size=None),
            ),
        )
    elif architecture == "resnet":
        network = ResNet(
            in_channels=1,
            slice_wise=mode == "slice",
            use_coordinate_channels=True,
            stem=StemConfig(out_channels=4, kernel_size=(3,) * dimensions, stride=(1,) * dimensions),
            stages=(ResStageConfig(out_channels=4, kernel_size=(3,) * dimensions, stride=(1,) * dimensions),),
        )
    else:
        network = ViT(
            in_channels=1,
            slice_wise=mode == "slice",
            use_coordinate_channels=True,
            patches=PatchConfig(input_size=(4, 8, 8) if dimensions == 3 else (8, 8), patch_size=(2,) * dimensions),
            encoder=EncoderConfig(num_layers=1, num_heads=2, hidden_dim=8, mlp_dim=16),
        )
    images = torch.rand((2, 1, 8, 8) if mode == "image" else (2, 1, 4, 8, 8))
    network.eval()
    with torch.no_grad():
        expected_logits = network(images)
        result = attribute(network, images)
    torch.testing.assert_close(result.logits, expected_logits)
    assert result.maps.shape == images.shape
    assert torch.isfinite(result.maps).all()
    assert result.maps.min() >= 0
    assert result.maps.max() > 0
    assert all(parameter.grad is None for parameter in network.parameters())


@pytest.mark.parametrize("method", ["saliency", "integrated_gradients", "occlusion"])
def test_volume_methods_preserve_state(method: Method) -> None:
    """Keep evaluation statistics, mixed module modes, and existing gradients."""
    torch.manual_seed(42)
    network = CNN(
        in_channels=1,
        stages=(CNNStageConfig(out_channels=2, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), pool_size=None),),
    )
    network.stages.eval()
    modes = [module.training for module in network.modules()]
    before = {name: value.clone() for name, value in network.state_dict().items()}
    for parameter in network.parameters():
        parameter.grad = torch.ones_like(parameter)
    result = attribute(network, torch.rand(2, 1, 3, 4, 5), AttributionConfig(method=method, steps=2, occlusion_size=2, occlusion_stride=3))
    assert result.maps.shape == (2, 1, 3, 4, 5)
    assert torch.isfinite(result.maps).all()
    assert modes == [module.training for module in network.modules()]
    for name, value in network.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert all(parameter.grad is not None and torch.equal(parameter.grad, torch.ones_like(parameter)) for parameter in network.parameters())


def test_gradcam_cleanup_and_frozen_backbone() -> None:
    """Remove hooks on failure and explain a fully frozen torchvision model."""
    network = LinearImageClassifier()
    with pytest.raises(ValueError, match="class count"):
        attribute(network, torch.ones(1, 1, 4, 4), AttributionConfig(target_layer="features", target_class=2))
    assert network.training
    assert not network.features._forward_hooks
    backbone = TorchvisionClassifier(pretrained=False, freeze_backbone=True)
    backbone.requires_grad_(False)
    result = attribute(backbone, torch.rand(1, 3, 32, 32))
    assert result.maps.shape == (1, 1, 32, 32)
    assert torch.isfinite(result.maps).all()
    assert backbone.training and not backbone.backbone.training


@pytest.mark.parametrize("input_kind", ["image", "npy"])
def test_explain_checkpoint_cli(input_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore a checkpoint and save raw arrays and labeled PNG panels."""
    parser = create_training_parser(TrainConfig)
    parsed = parser.parse_args(["--config", "configs/synthetic.yaml"])
    config, network = instantiate_training_config(parser, TrainConfig, parsed)
    module = ClassificationModule(config.training, network)
    checkpoint = tmp_path / "model.ckpt"
    torch.save(
        {"run_config": dump_training_config(parser, parsed, config), "state_dict": module.state_dict(), "class_names": ["negative", "positive"]},
        checkpoint,
    )
    input_path = tmp_path / ("input.png" if input_kind == "image" else "input.npy")
    if input_kind == "image":
        Image.new("RGB", (32, 32), (30, 80, 120)).save(input_path)
    else:
        np.save(input_path, np.full((3, 32, 32), 0.5, dtype=np.float32))
    destination = tmp_path / "explanation"
    monkeypatch.setattr(
        sys, "argv", ["computer-vision-explain", "--checkpoint", str(checkpoint), "--input", str(input_path), "--output_dir", str(destination)]
    )
    cli.main()
    with np.load(destination / "maps.npz", allow_pickle=False) as arrays:
        assert arrays["maps"].shape == (1, 1, 32, 32)
        assert arrays["class_names"].tolist() == ["negative", "positive"]
        assert arrays["logits"].shape == (1, 2)
    with Image.open(destination / "sample-000-depth-000.png") as panel:
        assert panel.size == (480, 80)


@pytest.mark.parametrize("stride", [1, 3])
def test_occlusion_overlap_and_edges(stride: int) -> None:
    """Cover odd spatial sizes and average overlapping window score drops."""
    images = torch.full((1, 1, 3, 5), 0.8)
    result = attribute(
        LinearImageClassifier(),
        images,
        AttributionConfig(method="occlusion", baseline=0.2, occlusion_size=2, occlusion_stride=stride),
    )
    torch.testing.assert_close(result.maps, torch.full_like(images, 2.4))


def test_signed_volume_display_scale(tmp_path: Path) -> None:
    """Keep weak attributions visible and preserve relative depth strength."""
    images = torch.full((1, 1, 2, 2, 2), 0.5)
    maps = torch.empty_like(images)
    maps[:, :, 0] = 5e-11
    maps[:, :, 1] = -1e-10
    result = AttributionResult(torch.zeros(1, 2), torch.tensor([1]), maps, maps)
    settings = AttributionConfig(method="integrated_gradients")
    save_visualizations(images, result, settings, tmp_path)
    with np.load(tmp_path / "maps.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["maps"], maps.numpy())
    with Image.open(tmp_path / "sample-000-depth-000.png") as positive:
        assert positive.getpixel((2, 48)) == (128, 0, 0)
    with Image.open(tmp_path / "sample-000-depth-001.png") as negative:
        assert negative.getpixel((2, 48)) == (0, 0, 255)
