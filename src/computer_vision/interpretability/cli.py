"""Explain a checkpoint prediction and save attribution visualizations."""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image

from computer_vision.config import ConfigModel, parse_and_validate
from computer_vision.data import image_transform
from computer_vision.evaluate import load_checkpoint
from computer_vision.interpretability.attribution import AttributionConfig, attribute
from computer_vision.interpretability.visualization import save_visualizations

LOGGER = logging.getLogger(__name__)


class InterpretationConfig(ConfigModel):
    """Checkpoint, input, and attribution output settings."""

    checkpoint: Path
    input: Path
    """Image file or .npy float array shaped (C, H, W) or (C, D, H, W)."""
    output_dir: Path
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    attribution: AttributionConfig = AttributionConfig()


def main() -> None:
    """Save a checkpoint's raw attributions and PNG overlays for one input."""
    logging.basicConfig(level=logging.INFO)
    arguments = parse_and_validate(InterpretationConfig)
    module, training_config = load_checkpoint(arguments.checkpoint)
    network = module.network.to(arguments.device)
    if arguments.input.suffix.lower() == ".npy":
        array = np.load(arguments.input, allow_pickle=False)
        if not np.issubdtype(array.dtype, np.floating) or array.ndim not in (3, 4):
            raise ValueError("NumPy inputs must be float arrays shaped (C, H, W) or (C, D, H, W)")
        images = torch.from_numpy(array.astype(np.float32)).unsqueeze(0)
    else:
        if network.in_channels != 3 or getattr(network, "slice_wise", False) or getattr(network, "dimensions", 2) == 3:
            raise ValueError("Use a .npy input for non-RGB, slice, or volume networks")
        with Image.open(arguments.input) as image:
            images = image_transform(training_config.dataset)(image.convert("RGB")).unsqueeze(0)
    images = images.to(arguments.device)
    result = attribute(network, images, arguments.attribution)
    save_visualizations(images, result, arguments.attribution, arguments.output_dir, module.class_names)
    LOGGER.info("Saved %s attribution arrays and overlays to %s", arguments.attribution.method, arguments.output_dir)


if __name__ == "__main__":
    main()
