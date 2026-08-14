"""Tests for shared inference-network serialization."""

from pathlib import Path

import numpy as np
import onnxruntime
import torch
from torch import nn

from computer_vision.artifacts import export_network_to_onnx, export_network_to_torch_export


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


