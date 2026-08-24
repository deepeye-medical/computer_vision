"""Loss functions for classification training."""

from typing import Annotated, Literal, TypeAlias

import torch
from pydantic import Field
from torch import nn
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss

from computer_vision.config import ConfigModel








class CrossEntropyConfig(ConfigModel):
    """Multiclass cross-entropy settings."""

    name: Literal["cross_entropy"] = "cross_entropy"
    label_smoothing: float = Field(default=0, ge=0, lt=1)


LossConfig: TypeAlias = Annotated[CrossEntropyConfig, Field(discriminator="name")]






