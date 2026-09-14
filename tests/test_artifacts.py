"""Tests for shared inference-network serialization."""

from pathlib import Path

import numpy as np
import onnxruntime
import pytest
import torch
from torch import nn

from computer_vision.artifacts import export_network_to_onnx, export_network_to_torch_export
from computer_vision.model import CNN, CNNStageConfig, ResNet, ResStageConfig, StemConfig


def test_torch_export_uses_inference_mode(tmp_path: Path) -> None:
    """Export an evaluation-mode network with a dynamic batch axis."""
    network = nn.Conv2d(1, 2, kernel_size=1)
    network.train()
    example_input = torch.rand(1, 1, 8, 8)
    output_path = tmp_path / "network.pt2"

    export_network_to_torch_export(network, example_input, output_path)
    exported = torch.export.load(output_path)

    assert not network.training
    assert exported.module()(torch.rand(3, 1, 8, 8)).shape == (3, 2, 8, 8)


def test_onnx_export_preserves_tensor_names_and_logits(tmp_path: Path) -> None:
    """Export named logits with a dynamic ONNX batch axis."""
    network = nn.Conv2d(1, 2, kernel_size=1)
    example_input = torch.rand(1, 1, 8, 8)
    output_path = tmp_path / "network.onnx"

    export_network_to_onnx(network, example_input, output_path, input_name="images")
    session = onnxruntime.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])

    assert session.get_inputs()[0].name == "images"
    assert session.get_outputs()[0].name == "logits"
    input_batch = np.random.default_rng(0).random((3, 1, 8, 8), dtype=np.float32)
    output = np.asarray(session.run(None, {"images": input_batch})[0])
    assert output.shape == (3, 2, 8, 8)


def test_resnet_onnx_matches_pytorch(tmp_path: Path) -> None:
    """Preserve residual and coordinate-channel predictions across batch sizes."""
    from computer_vision.model import ResNet, ResStageConfig

    torch.manual_seed(42)
    network = ResNet(stages=(ResStageConfig(out_channels=4),), use_coordinate_channels=True).eval()
    output_path = tmp_path / "resnet.onnx"
    export_network_to_onnx(network, torch.rand(2, 3, 16, 16), output_path, input_name="images")
    session = onnxruntime.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    images = torch.rand(3, 3, 16, 16)
    with torch.no_grad():
        expected = network(images).numpy()
    actual = np.asarray(session.run(None, {"images": images.numpy()})[0])
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "architecture,mode", [("cnn", "image"), ("cnn", "slice"), ("cnn", "volume"), ("resnet", "slice"), ("resnet", "volume"), ("resnet", "factorized")]
)
def test_volume_and_slice_exports(architecture: str, mode: str, tmp_path: Path) -> None:
    """Preserve logits and signed coordinates in exports with dynamic batches."""
    torch.manual_seed(42)
    torch.set_num_threads(1)
    dimensions = 3 if mode in ("volume", "factorized") else 2
    if architecture == "cnn":
        network = CNN(
            in_channels=1,
            stages=(
                CNNStageConfig(out_channels=4, kernel_size=(3,) * dimensions, stride=(1,) * dimensions, padding=(0,) * dimensions, pool_size=None),
            ),
            slice_wise=mode == "slice",
            slice_pooling="mean_max",
            use_coordinate_channels=True,
        )
    else:
        network = ResNet(
            in_channels=1,
            stem=StemConfig(
                out_channels=4, kernel_size=(3,) * dimensions, stride=(1,) * dimensions, intermediate_channels=3 if mode == "factorized" else None
            ),
            stages=(ResStageConfig(out_channels=4, kernel_size=(3,) * dimensions, stride=(1,) * dimensions),),
            block_type="factorized" if mode == "factorized" else "standard",
            slice_wise=mode == "slice",
            slice_pooling="mean_max",
            use_coordinate_channels=True,
        )
    example_shape = (2, 1, 8, 8) if mode == "image" else (2, 1, 4, 8, 8)
    example = torch.rand(example_shape)
    onnx_path = tmp_path / "network.onnx"
    exported_path = tmp_path / "network.pt2"
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
