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
