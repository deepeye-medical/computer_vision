"""Export checkpoint networks for inference."""

from pathlib import Path
from typing import Annotated, Literal

import torch
from pydantic import Field

from computer_vision.artifacts import export_network_to_onnx, export_network_to_torch_export, save_network_state_dict
from computer_vision.config import ConfigModel, parse_and_validate
from computer_vision.evaluate import load_checkpoint


class ExportConfig(ConfigModel):
    """Checkpoint and destination for an inference artifact."""

    checkpoint: Path
    output: Path
    format: Literal["onnx", "torch_export", "weights"] = "onnx"
    input_shape: tuple[Annotated[int, Field(gt=0)], ...] | None = None
    """Fixed input shape without the batch axis; required for slice exports."""


def main() -> None:
    """Export logits with a dynamic batch axis and fixed spatial dimensions."""
    arguments = parse_and_validate(ExportConfig)
    module, config = load_checkpoint(arguments.checkpoint)
    network = module.network.cpu().eval()
    if arguments.format == "weights":
        save_network_state_dict(network, arguments.output)
        return
    input_shape = arguments.input_shape
    if input_shape is None:
        if getattr(network, "slice_wise", False):
            raise ValueError("Slice-wise exports require input_shape=[channels, slices, height, width]")
        spatial_size = getattr(network, "input_size", None)
        if spatial_size is None:
            if getattr(network, "dimensions", 2) == 3:
                raise ValueError("Volume exports require input_shape=[channels, depth, height, width]")
            spatial_size = (config.dataset.image_size, config.dataset.image_size)
        input_shape = (network.in_channels, *spatial_size)
    example = torch.zeros(2, *input_shape)
    if arguments.format == "onnx":
        export_network_to_onnx(network, example, arguments.output, input_name="images")
    elif arguments.format == "torch_export":
        export_network_to_torch_export(network, example, arguments.output)


if __name__ == "__main__":
    main()
