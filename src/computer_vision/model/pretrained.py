"""Torchvision backbones with task heads and a shared RGB input contract."""

from typing import Literal, Self

import torch
from torch import nn
from torchvision import models

from computer_vision.model.base import VisionNetwork


class TorchvisionClassifier(VisionNetwork):
    """Fine-tune a standard ResNet or vision transformer."""

    input_mean: torch.Tensor
    input_std: torch.Tensor

    def __init__(
        self,
        backbone: Literal["resnet18", "resnet50", "vit_b_16"] = "resnet18",
        in_channels: int = 3,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        """Load a backbone and replace its ImageNet classification head.

        Parameters
        ----------
        backbone
            Torchvision architecture to load.
        in_channels
            RGB input channels; these backbones require three.
        num_classes
            Number of task classes.
        pretrained
            Load ImageNet-1K V1 weights. Downloads occur only when not cached.
        freeze_backbone
            Train only the new head and keep backbone running statistics fixed.
        """
        super().__init__(in_channels)
        if in_channels != 3 or num_classes < 2:
            raise ValueError("Torchvision classifiers require RGB inputs and at least two classes")
        weights = models.get_model_weights(backbone)["IMAGENET1K_V1"]
        model = models.get_model(backbone, weights=weights if pretrained else None)
        if isinstance(model, models.ResNet):
            head = nn.Linear(model.fc.in_features, num_classes)
            model.fc = head
            self.image_size: int | None = None
        elif isinstance(model, models.VisionTransformer):
            head = nn.Linear(model.hidden_dim, num_classes)
            model.heads = nn.Sequential(head)
            self.image_size = model.image_size
        else:
            raise ValueError("Unsupported torchvision backbone")
        if freeze_backbone:
            model.requires_grad_(False)
            head.requires_grad_(True)
        self.backbone = model
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        preprocessing = weights.transforms()
        self.register_buffer("input_mean", torch.tensor(preprocessing.mean).reshape(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor(preprocessing.std).reshape(1, 3, 1, 1))
        self.train()

    def train(self, mode: bool = True) -> Self:
        """Keep frozen backbone statistics and dropout fixed during head training."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize RGB tensors in [0, 1] and return task logits."""
        return self.backbone((images - self.input_mean) / self.input_std)
