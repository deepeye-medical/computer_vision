"""Check transformer input modes, pretrained transfer, checkpoints, and exports."""

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

import computer_vision.export as export
from computer_vision.artifacts import export_network_to_onnx, export_network_to_torch_export
from computer_vision.evaluate import load_checkpoint
from computer_vision.lightning_module import ClassificationModule
from computer_vision.model import EncoderConfig, PatchConfig, ViT
from computer_vision.train import TrainConfig, create_training_parser, dump_training_config, instantiate_training_config


@pytest.fixture(autouse=True)
def seed_transformer() -> None:
    """Keep compact transformer tests deterministic and fast on CPU."""
    torch.manual_seed(42)
    torch.set_num_threads(1)


@pytest.mark.parametrize("mode,pooling", [("image", "cls"), ("slice", "mean"), ("volume", "mean_max")])
def test_vit_modes_train(mode: str, pooling: Literal["cls", "mean", "mean_max"]) -> None:
    """Train every input layout with coordinates and all token pooling choices."""
    volume = mode == "volume"
    network = ViT(
        in_channels=1,
        patches=PatchConfig(input_size=(4, 16, 16) if volume else (16, 16), patch_size=(2, 8, 8) if volume else (8, 8)),
        encoder=EncoderConfig(num_layers=1, num_heads=2, hidden_dim=16, mlp_dim=32),
        slice_wise=mode == "slice",
        pooling=pooling,
        slice_pooling="mean_max",
        use_coordinate_channels=True,
    )
    images = torch.rand((1, 1, 16, 16) if mode == "image" else (1, 1, 4, 16, 16))
    loss = F.cross_entropy(network(images), torch.tensor([1]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.requires_grad and parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in network.parameters())
    assert network.encoder.pos_embedding.shape[1] == (9 if volume else 5)


def test_slice_vit_encodes_original_positions_independently() -> None:
    """Join B-scan features after independent attention, keeping depth coordinates."""
    network = ViT(
        in_channels=1,
        patches=PatchConfig(input_size=(16, 16), patch_size=(8, 8)),
        encoder=EncoderConfig(num_layers=1, num_heads=2, hidden_dim=16, mlp_dim=32),
        slice_wise=True,
        use_coordinate_channels=True,
        dropout=0,
    ).eval()
    images = torch.rand(2, 1, 3, 16, 16)
    patch_inputs: list[torch.Tensor] = []

    def record_input(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        patch_inputs.append(inputs[0].detach())

    hook = network.patch_embedding.register_forward_pre_hook(record_input)
    with torch.no_grad():
        actual = network(images)
    hook.remove()
    torch.testing.assert_close(patch_inputs[0][:, -1, 0, 0], torch.tensor([-1, 0, 1, -1, 0, 1], dtype=images.dtype))
    with torch.no_grad():
        tokens = network.patch_embedding(patch_inputs[0])
        encoded = network.encoder(torch.cat((network.class_token.expand(6, -1, -1), tokens), dim=1))
        torch.testing.assert_close(actual, network.classifier(encoded[:, 0].reshape(2, 3, -1).mean(dim=1)))


@pytest.mark.parametrize("legacy", [False, True])
def test_image_pretraining_matches_torchvision(legacy: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve encoder output when loading current or original MLP key names."""
    source = models.VisionTransformer(image_size=224, patch_size=16, num_layers=1, num_heads=2, hidden_dim=16, mlp_dim=32).eval()
    source_state = source.state_dict()
    if legacy:
        source_state = {name.replace(".mlp.0.", ".mlp.linear_1.").replace(".mlp.3.", ".mlp.linear_2."): value for name, value in source_state.items()}

    def load_weights(*_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        return source_state

    monkeypatch.setattr(models.ViT_B_16_Weights, "get_state_dict", load_weights)
    network = ViT(encoder=EncoderConfig(num_layers=1, num_heads=2, hidden_dim=16, mlp_dim=32), weights="vit_b_16_imagenet1k_v1", dropout=0).eval()
    network.load_pretrained_weights()
    source.heads = nn.Sequential(nn.Identity())
    images = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        normalized = (images - network.input_mean.reshape(1, 3, 1, 1)) / network.input_std.reshape(1, 3, 1, 1)
        torch.testing.assert_close(network(images), network.classifier(source(normalized)))


@pytest.mark.parametrize("mode", ["image", "slice", "volume"])
def test_pretrained_adaptation_and_checkpoint(mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Adapt channels, patch depth, and position grids, then restore without downloads."""
    volume = mode == "volume"
    parser = create_training_parser(TrainConfig)
    parsed = parser.parse_object(
        {
            "dataset": {"image_size": 32},
            "network": {
                "class_path": "computer_vision.model.ViT",
                "init_args": {
                    "in_channels": 1,
                    "patches": {"input_size": [4, 32, 32] if volume else [32, 32], "patch_size": [2, 16, 16] if volume else [16, 16]},
                    "encoder": {"num_layers": 1, "num_heads": 2, "hidden_dim": 16, "mlp_dim": 32},
                    "weights": "vit_b_16_imagenet1k_v1",
                    "slice_wise": mode == "slice",
                    "use_coordinate_channels": True,
                },
            },
        }
    )
    source = models.VisionTransformer(image_size=224, patch_size=16, num_layers=1, num_heads=2, hidden_dim=16, mlp_dim=32)
    source_state = source.state_dict()

    def load_weights(*_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        return source_state

    monkeypatch.setattr(models.ViT_B_16_Weights, "get_state_dict", load_weights)
    config, network = instantiate_training_config(parser, TrainConfig, parsed)
    assert isinstance(network, ViT)
    projection = network.patch_embedding.projection.weight
    assert torch.count_nonzero(projection[:, 1:]) == 0
    spatial_projection = projection[:, :1].sum(dim=2) if volume else projection[:, :1]
    torch.testing.assert_close(spatial_projection, source_state["conv_proj.weight"].sum(dim=1, keepdim=True))
    source_positions = source_state["encoder.pos_embedding"]
    spatial_positions = (
        F.interpolate(source_positions[:, 1:].transpose(1, 2).reshape(1, 16, 14, 14), size=(2, 2), mode="bicubic", align_corners=True)
        .flatten(2)
        .transpose(1, 2)
    )
    if volume:
        spatial_positions = spatial_positions.repeat(1, 2, 1)
    torch.testing.assert_close(network.encoder.pos_embedding, torch.cat((source_positions[:, :1], spatial_positions), dim=1))
    module = ClassificationModule(config.training, network).eval()
    checkpoint = tmp_path / "model.ckpt"
    torch.save(
        {"run_config": dump_training_config(parser, parsed, config), "state_dict": module.state_dict(), "class_names": ["negative", "positive"]},
        checkpoint,
    )

    def forbid_download(*_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        pytest.fail("Checkpoint restoration and resumes must not download weights")

    monkeypatch.setattr(models.ViT_B_16_Weights, "get_state_dict", forbid_download)
    restored, restored_config = load_checkpoint(checkpoint)
    restored.eval()
    images = torch.rand((1, 1, 32, 32) if mode == "image" else (1, 1, 4, 32, 32))
    with torch.no_grad():
        torch.testing.assert_close(restored(images), module(images))
    assert restored_config == config
    exported_path = tmp_path / "checkpoint.pt2"
    export_args = ["computer-vision-export", "--checkpoint", str(checkpoint), "--output", str(exported_path), "--format", "torch_export"]
    if mode == "slice":
        export_args.extend(["--input_shape", "[1, 4, 32, 32]"])
    monkeypatch.setattr(sys, "argv", export_args)
    export.main()
    with torch.no_grad():
        torch.testing.assert_close(torch.export.load(exported_path).module()(images), module(images))
    parsed.resume_from = checkpoint
    _, resumed = instantiate_training_config(parser, TrainConfig, parsed)
    assert isinstance(resumed, ViT)
    torch.testing.assert_close(resumed.input_mean, network.input_mean)


@pytest.mark.parametrize("mode", ["image", "slice", "volume"])
def test_vit_exports(mode: str, tmp_path: Path) -> None:
    """Preserve logits in ONNX and torch.export with dynamic batch sizes."""
    volume = mode == "volume"
    network = ViT(
        in_channels=1,
        patches=PatchConfig(input_size=(4, 16, 16) if volume else (16, 16), patch_size=(2, 8, 8) if volume else (8, 8)),
        encoder=EncoderConfig(num_layers=1, num_heads=2, hidden_dim=16, mlp_dim=32),
        slice_wise=mode == "slice",
        pooling="mean_max",
        slice_pooling="mean_max",
        use_coordinate_channels=True,
    )
    example_shape = (2, 1, 16, 16) if mode == "image" else (2, 1, 4, 16, 16)
    onnx_path = tmp_path / "vit.onnx"
    exported_path = tmp_path / "vit.pt2"
    example = torch.rand(example_shape)
    export_network_to_onnx(network, example, onnx_path, input_name="images")
    export_network_to_torch_export(network, example, exported_path)
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    exported = torch.export.load(exported_path).module()
    for batch_size in (1, 3):
        images = torch.rand(batch_size, *example_shape[1:])
        with torch.no_grad():
            expected = network(images)
            torch.testing.assert_close(exported(images), expected)
        actual = np.asarray(session.run(None, {"images": images.numpy()})[0])
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-4, atol=1e-5)


def test_pretrained_loading_rejects_partial_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail instead of silently loading only the first block of a larger encoder."""
    source = models.VisionTransformer(image_size=224, patch_size=16, num_layers=2, num_heads=2, hidden_dim=16, mlp_dim=32).state_dict()

    def load_weights(*_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        return source

    monkeypatch.setattr(models.ViT_B_16_Weights, "get_state_dict", load_weights)
    network = ViT(encoder=EncoderConfig(num_layers=1, num_heads=2, hidden_dim=16, mlp_dim=32), weights="vit_b_16_imagenet1k_v1")
    with pytest.raises(ValueError, match="does not use pretrained parameters"):
        network.load_pretrained_weights()
