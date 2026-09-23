"""Check stage-based image, slice, and volume classification."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from computer_vision.evaluate import load_checkpoint
from computer_vision.lightning_module import ClassificationModule
from computer_vision.model import ResBlock, ResNet, ResStage, ResStageConfig, ResStem, StemConfig
from computer_vision.model.resnet import PRETRAINED_WEIGHTS
from computer_vision.train import TrainConfig, create_training_parser, dump_training_config, instantiate_training_config


@pytest.mark.parametrize("mode", ["image", "slice", "volume", "factorized"])
def test_configured_stages_train(mode: str) -> None:
    """Backpropagate through each mode, including coordinates and slice pooling."""
    torch.manual_seed(42)
    torch.set_num_threads(1)
    dimensions = 3 if mode in ("volume", "factorized") else 2
    kernel = (3,) * dimensions
    stem = StemConfig(out_channels=8, kernel_size=kernel, stride=(1,) * dimensions, intermediate_channels=6 if mode == "factorized" else None)
    network = ResNet(
        in_channels=1,
        stem=stem,
        stages=(ResStageConfig(out_channels=8, num_blocks=2, kernel_size=kernel, stride=(2,) * dimensions),),
        block_type="factorized" if mode == "factorized" else "standard",
        slice_wise=mode == "slice",
        slice_pooling="mean_max",
        use_coordinate_channels=True,
    )
    images = torch.randn((1, 1, 16, 16) if mode == "image" else (1, 1, 4, 16, 16))
    loss = F.cross_entropy(network(images), torch.tensor([1]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in network.parameters())
    assert isinstance(network.stem, ResStem)
    stage = network.stages[0]
    assert isinstance(stage, ResStage)
    assert len(stage) == 2
    if mode != "factorized":
        first, second = stage
        assert isinstance(first, ResBlock) and isinstance(second, ResBlock)
        assert isinstance(first.downsample, nn.Sequential)
        assert isinstance(second.downsample, nn.Identity)


def test_slice_processing_preserves_depth_coordinates() -> None:
    """Encode original slice positions before the shared 2D backbone sees them."""
    network = ResNet(
        in_channels=1,
        stem=StemConfig(out_channels=4),
        stages=(ResStageConfig(out_channels=4),),
        slice_wise=True,
        use_coordinate_channels=True,
        pooling="mean",
        dropout=0,
    ).eval()
    volumes = torch.rand(2, 1, 3, 8, 8)
    stem_inputs: list[torch.Tensor] = []

    def record_input(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        stem_inputs.append(inputs[0].detach())

    hook = network.stem.register_forward_pre_hook(record_input)
    with torch.no_grad():
        actual = network(volumes)
    hook.remove()
    torch.testing.assert_close(stem_inputs[0][:, -1, 0, 0], torch.tensor([-1, 0, 1, -1, 0, 1], dtype=volumes.dtype))
    with torch.no_grad():
        encoded = network.stages(network.stem(stem_inputs[0])).mean(dim=(2, 3)).reshape(2, 3, -1)
        torch.testing.assert_close(actual, network.classifier(encoded.mean(dim=1)))
    singleton = network.append_coordinate_channels(volumes[:, :, :1])
    assert torch.count_nonzero(singleton[:, -1]) == 0


@pytest.mark.parametrize(
    "overlay",
    [
        "model_imagenet_resnet",
        "model_resnet34_image",
        "model_resnet18_slice",
        "model_resnet34_slice",
        "model_resnet18_volume",
        "model_resnet34_volume",
        "model_r2plus1d_18",
    ],
)
def test_pretrained_transfer(overlay: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Match source features or inflated kernels without downloading checkpoints."""
    torch.manual_seed(42)
    torch.set_num_threads(1)
    parser = create_training_parser(TrainConfig)
    parsed = parser.parse_args(["--config", "configs/synthetic.yaml", "--config", f"configs/{overlay}.yaml", "--network.init_args.dropout", "0"])
    _, network = instantiate_training_config(parser, TrainConfig, parsed, initialize_pretrained=False)
    assert isinstance(network, ResNet)
    assert network.weights is not None
    if "r2plus1d" in overlay:
        source = models.video.r2plus1d_18(weights=None)
    elif "34" in overlay:
        source = models.resnet34(weights=None)
    else:
        source = models.resnet18(weights=None)
    source_state = source.state_dict()

    def load_weights(*_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        return source_state

    monkeypatch.setattr(type(PRETRAINED_WEIGHTS[network.weights]), "get_state_dict", load_weights)
    network.load_pretrained_weights()
    assert all(parameter.requires_grad for parameter in network.parameters())
    if network.block_type == "standard" and network.dimensions == 3:
        torch.testing.assert_close(network.state_dict()["stem.0.weight"].sum(dim=2), source_state["conv1.weight"])
        torch.testing.assert_close(network.state_dict()["stages.0.0.conv1.weight"].sum(dim=2), source_state["layer1.0.conv1.weight"])
    else:
        network.eval()
        source.eval()
        source.fc = nn.Identity()
        images = torch.rand(2, 3, 4, 32, 32) if network.slice_wise or network.dimensions == 3 else torch.rand(2, 3, 32, 32)
        normalized = (images - network.input_mean.reshape(1, 3, *([1] * (images.ndim - 2)))) / network.input_std.reshape(
            1, 3, *([1] * (images.ndim - 2))
        )
        with torch.no_grad():
            if network.slice_wise:
                features = source(normalized.permute(0, 2, 1, 3, 4).reshape(8, 3, 32, 32)).reshape(2, 4, -1).mean(dim=1)
            else:
                features = source(normalized)
            torch.testing.assert_close(network(images), network.classifier(features))


@pytest.mark.parametrize("overlay", ["model_resnet18_volume", "model_r2plus1d_18"])
def test_coordinate_transfer_and_checkpoint(overlay: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Load grayscale and zero-coordinate filters, then restore without downloads."""
    torch.manual_seed(42)
    parser = create_training_parser(TrainConfig)
    parsed = parser.parse_args(
        [
            "--config",
            "configs/synthetic.yaml",
            "--config",
            f"configs/{overlay}.yaml",
            "--network.init_args.in_channels",
            "1",
            "--network.init_args.use_coordinate_channels",
            "true",
        ]
    )
    _, unloaded = instantiate_training_config(parser, TrainConfig, parsed, initialize_pretrained=False)
    assert isinstance(unloaded, ResNet)
    assert unloaded.weights is not None
    source = models.video.r2plus1d_18(weights=None) if "r2plus1d" in overlay else models.resnet18(weights=None)
    source_state = source.state_dict()

    def load_weights(*_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        return source_state

    monkeypatch.setattr(type(PRETRAINED_WEIGHTS[unloaded.weights]), "get_state_dict", load_weights)
    config, network = instantiate_training_config(parser, TrainConfig, parsed)
    assert isinstance(network, ResNet)
    assert network.weights is not None
    stem_weight = network.state_dict()["stem.0.weight"]
    assert torch.count_nonzero(stem_weight[:, 1:]) == 0
    source_name = "stem.0.weight" if "r2plus1d" in overlay else "conv1.weight"
    expected = source_state[source_name].sum(dim=1, keepdim=True)
    actual = stem_weight[:, :1] if "r2plus1d" in overlay else stem_weight[:, :1].sum(dim=2)
    torch.testing.assert_close(actual, expected)
    module = ClassificationModule(config.training, network).eval()
    checkpoint = tmp_path / "model.ckpt"
    torch.save(
        {"run_config": dump_training_config(parser, parsed, config), "state_dict": module.state_dict(), "class_names": ["negative", "positive"]},
        checkpoint,
    )

    def forbid_download(*_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        pytest.fail("Checkpoint restoration must not download weights")

    monkeypatch.setattr(type(PRETRAINED_WEIGHTS[network.weights]), "get_state_dict", forbid_download)
    restored, restored_config = load_checkpoint(checkpoint)
    restored.eval()
    images = torch.rand(1, 1, 8, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(restored(images), module(images))
    assert restored_config == config
    parsed.resume_from = checkpoint
    _, resumed = instantiate_training_config(parser, TrainConfig, parsed)
    assert isinstance(resumed, ResNet)
    torch.testing.assert_close(resumed.input_mean, network.input_mean)


@pytest.mark.parametrize("slice_wise", [True, False])
def test_incompatible_configuration_fails(slice_wise: bool) -> None:
    """Reject 3D slice encoders and incompatible pretrained stage layouts."""
    with pytest.raises(ValueError, match=r"slice_wise|stage counts"):
        if slice_wise:
            ResNet(
                stem=StemConfig(kernel_size=(3, 3, 3), stride=(1, 1, 1)),
                stages=(ResStageConfig(out_channels=4, kernel_size=(3, 3, 3), stride=(1, 1, 1)),),
                slice_wise=True,
            )
        else:
            ResNet(weights="resnet18_imagenet1k_v1")
