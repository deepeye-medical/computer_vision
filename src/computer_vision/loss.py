"""Loss functions for classification training."""

from typing import Annotated, Literal, TypeAlias

import torch
from pydantic import Field
from torch import nn
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss

from computer_vision.config import ConfigModel


class BCEConfig(ConfigModel):
    """Binary cross-entropy settings."""

    name: Literal["bce"] = "bce"
    positive_class_weight: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    """Optional positive-class weight used by binary cross entropy."""


class FocalConfig(ConfigModel):
    """Binary focal-loss settings."""

    name: Literal["focal"] = "focal"
    alpha: float = Field(default=0.25, ge=0.0, le=1.0)
    """Positive-class weighting factor."""
    gamma: float = Field(default=2.0, ge=0.0, allow_inf_nan=False)
    """Hard-example focusing exponent."""


class GCEConfig(ConfigModel):
    """Binary generalized cross-entropy settings."""

    name: Literal["gce"] = "gce"
    q: float = Field(default=0.7, gt=0.0, le=1.0)
    """Robustness exponent; near zero approaches BCE, one gives absolute error."""


class CrossEntropyConfig(ConfigModel):
    """Multiclass cross-entropy settings."""

    name: Literal["cross_entropy"] = "cross_entropy"
    label_smoothing: float = Field(default=0, ge=0, lt=1)


LossConfig: TypeAlias = Annotated[CrossEntropyConfig | BCEConfig | FocalConfig | GCEConfig, Field(discriminator="name")]


class GeneralizedCrossEntropyLoss(nn.Module):
    """Binary generalized cross entropy operating on logits and hard labels."""

    def __init__(self, q: float) -> None:
        """Initialize the noise-robust loss.

        Parameters
        ----------
        q
            Robustness exponent in (0, 1]. Near zero the loss approaches BCE;
            at one it equals the absolute error in the positive probability.
        """
        super().__init__()
        self.q = q

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return mean GCE, using float32 and expm1 for numerical stability."""
        logits = logits.float()
        targets = targets.to(logits).reshape_as(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (-torch.expm1(-self.q * bce) / self.q).mean()




class FocalLoss(nn.Module):
    """Binary focal loss operating on logits."""

    def __init__(self, alpha: float, gamma: float) -> None:
        """Initialize the focal loss.

        Parameters
        ----------
        alpha
            Positive-class weighting factor.
        gamma
            Hard-example focusing exponent.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return the mean sigmoid focal loss."""
        return sigmoid_focal_loss(
            logits,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction="mean",
        )
