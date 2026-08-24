"""Shared vision network contract."""

from abc import ABC, abstractmethod
from typing import Annotated

import torch
from pydantic import Field
from torch import nn

Dropout = Annotated[float, Field(ge=0.0, le=1.0)]


class VisionNetwork(nn.Module, ABC):
    """Base for vision networks that accept the configured input channels."""

    def __init__(self, in_channels: int) -> None:
        """Store the shared image input contract.

        Parameters
        ----------
        in_channels
            Number of stored channels in each image.
        """
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be greater than zero")
        self.in_channels = in_channels

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits for each input image."""

    @staticmethod
    def append_coordinate_channels(images: torch.Tensor) -> torch.Tensor:
        """Append signed coordinates in width, height, then depth order.

        Coordinates describe the augmented tensor grid in [-1, 1], rather
        than physical distances. A singleton axis receives coordinate zero.
        Slice-wise callers append coordinates before splitting B-scans.
        """
        spatial_shape = images.shape[2:]
        coordinates = []
        for axis in reversed(range(len(spatial_shape))):
            size = spatial_shape[axis]
            shape = [1] * images.ndim
            shape[axis + 2] = size
            coordinate = torch.linspace(-1, 1, size, device=images.device, dtype=images.dtype) if size > 1 else images.new_zeros(1)
            coordinates.append(coordinate.reshape(shape).expand(images.shape[0], 1, *spatial_shape))
        return torch.cat((images, *coordinates), dim=1)
