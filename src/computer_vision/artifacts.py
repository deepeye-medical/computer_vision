"""Serialization helpers for bare PyTorch inference networks."""

from pathlib import Path

import torch
from torch import nn


def _prepare_output(path: Path) -> None:
    """Create the parent directory for an output artifact.

    Parameters
    ----------
    path
        Destination artifact path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def save_network_state_dict(network: nn.Module, output_path: Path) -> None:
    """Save weights whose keys belong directly to a bare network.

    Parameters
    ----------
    network
        Inference network whose state is serialized.
    output_path
        Destination for the serialized state dictionary.
    """
    _prepare_output(output_path)
    torch.save(network.state_dict(), output_path)


def export_network_to_torch_export(network: nn.Module, example_input: torch.Tensor, output_path: Path) -> None:
    """Save an inference-mode network as a PyTorch ExportedProgram.

    Parameters
    ----------
    network
        Network placed in evaluation mode before export.
    example_input
        Representative network input used to trace the computation.
    output_path
        Destination for the exported program.
    """
    _prepare_output(output_path)
    network.eval()
    repeat = (2,) + (1,) * (example_input.ndim - 1)
    export_input = example_input.repeat(repeat) if example_input.shape[0] == 1 else example_input
    dynamic_batch = torch.export.Dim("batch", min=1)
    with torch.inference_mode():
        exported_program = torch.export.export(
            network,
            (export_input,),
            dynamic_shapes=({0: dynamic_batch},),
        )
        torch.export.save(exported_program, output_path)


def export_network_to_onnx(
    network: nn.Module,
    example_input: torch.Tensor,
    output_path: Path,
    *,
    input_name: str,
    output_name: str = "logits",
) -> None:
    """Save an inference-mode network using PyTorch's ONNX exporter.

    Parameters
    ----------
    network
        Network placed in evaluation mode before export.
    example_input
        Representative network input used to trace the computation.
    output_path
        Destination for the ONNX model.
    input_name
        Name assigned to the exported input tensor.
    output_name
        Name assigned to the exported logits tensor.
    """
    _prepare_output(output_path)
    network.eval()
    repeat = (2,) + (1,) * (example_input.ndim - 1)
    export_input = example_input.repeat(repeat) if example_input.shape[0] == 1 else example_input
    dynamic_batch = torch.export.Dim("batch", min=1)
    with torch.inference_mode():
        torch.onnx.export(
            network,
            (export_input,),
            output_path,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_shapes=({0: dynamic_batch},),
        )
