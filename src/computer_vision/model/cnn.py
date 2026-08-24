"""Configurable CNN for images, independent slices, and volumes."""

from collections.abc import Callable
from typing import Annotated, Literal, cast

import torch
from jsonargparse.typing import register_type
from pydantic import Field, model_validator
from torch import nn

from computer_vision.config import ConfigModel
from computer_vision.model.base import Dropout, VisionNetwork

KernelSize = tuple[Annotated[int, Field(gt=0)], ...]
Padding = tuple[Annotated[int, Field(ge=0)], ...]
Pooling = Literal["mean", "mean_max"]


class CNNStageConfig(ConfigModel):
    """Configure one convolution, activation, normalization, and pooling stage."""

    out_channels: int = Field(default=32, gt=0)
    """Number of output feature channels."""
    kernel_size: KernelSize = (3, 3)
    """Convolution kernel sizes in spatial tensor order."""
    stride: KernelSize = (1, 1)
    """Convolution strides in spatial tensor order."""
    padding: Padding = (0, 0)
    """Padding on each side of the spatial axes."""
    pool_size: KernelSize | None = (2, 2)
    """Max-pooling kernel and stride; null disables pooling."""

    @model_validator(mode="after")
    def validate_dimensions(self) -> "CNNStageConfig":
        """Require matching image or volume axes for each operation."""
        dimensions = len(self.kernel_size)
        if dimensions not in (2, 3) or any(size is not None and len(size) != dimensions for size in (self.stride, self.padding, self.pool_size)):
            raise ValueError("CNN kernels, strides, padding, and pooling must use two or three matching axes")
        return self


register_type(CNNStageConfig, serializer=CNNStageConfig.model_dump, deserializer=CNNStageConfig.model_validate)


class CNNStage(nn.Sequential):
    """Convolution, ReLU, BatchNorm, and optional max pooling."""

    def __init__(self, in_channels: int, config: CNNStageConfig) -> None:
        """Build a convolution, activation, normalization, and pooling stage.

        Parameters
        ----------
        in_channels
            Number of channels entering the stage.
        config
            Convolution and optional max-pooling settings.
        """
        dimensions = len(config.kernel_size)
        convolution = cast(Callable[..., nn.Module], nn.Conv2d if dimensions == 2 else nn.Conv3d)
        normalization = nn.BatchNorm2d if dimensions == 2 else nn.BatchNorm3d
        layers: list[nn.Module] = [
            convolution(in_channels, config.out_channels, config.kernel_size, stride=config.stride, padding=config.padding),
            nn.ReLU(inplace=True),
            normalization(config.out_channels),
        ]
        if config.pool_size is not None:
            pooling_class = nn.MaxPool2d if dimensions == 2 else nn.MaxPool3d
            layers.append(pooling_class(config.pool_size))
        super().__init__(*layers)


class CNN(VisionNetwork):
    """Classify images, independent B-scans, or volumes with a small CNN."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        stages: tuple[CNNStageConfig, ...] = (CNNStageConfig(), CNNStageConfig(), CNNStageConfig(pool_size=(4, 4))),
        slice_wise: bool = False,
        pooling: Pooling = "mean",
        slice_pooling: Pooling = "mean",
        hidden_channels: int = 64,
        dropout: Dropout = 0.2,
        use_coordinate_channels: bool = False,
    ) -> None:
        """Build the convolutional stages and classifier.

        Parameters
        ----------
        in_channels
            Number of stored input channels, excluding coordinates.
        num_classes
            Number of task classes.
        stages
            Convolution and pooling stages in execution order.
        slice_wise
            Process each B-scan independently using shared 2D convolutions.
        pooling
            Spatial feature pooling before the head or slice aggregation.
        slice_pooling
            Feature pooling across independently processed B-scans.
        hidden_channels
            Number of features in the classifier's hidden layer.
        dropout
            Classifier dropout probability.
        use_coordinate_channels
            Append signed coordinates after augmentation. Slice inputs retain
            their B-scan position before independent feature extraction.
        """
        super().__init__(in_channels)
        if num_classes < 2 or hidden_channels < 1 or not stages:
            raise ValueError("Require at least two classes, positive hidden_channels, and one CNN stage")
        dimensions = len(stages[0].kernel_size)
        if any(len(stage.kernel_size) != dimensions for stage in stages):
            raise ValueError("All CNN stages must have the same dimensionality")
        if slice_wise and dimensions != 2:
            raise ValueError("slice_wise requires 2D kernels")
        self.dimensions = dimensions
        self.num_classes = num_classes
        self.slice_wise = slice_wise
        self.pooling = pooling
        self.slice_pooling = slice_pooling
        self.use_coordinate_channels = use_coordinate_channels
        coordinate_channels = (3 if slice_wise else dimensions) if use_coordinate_channels else 0
        previous_channels = in_channels + coordinate_channels
        feature_layers: list[nn.Module] = []
        for stage in stages:
            feature_layers.append(CNNStage(previous_channels, stage))
            previous_channels = stage.out_channels
        self.stages = nn.Sequential(*feature_layers)
        classifier_channels = previous_channels * (2 if pooling == "mean_max" else 1) * (2 if slice_wise and slice_pooling == "mean_max" else 1)
        # Feature normalization stays before pooling so single-volume batches work.
        self.classifier = nn.Sequential(
            nn.Linear(classifier_channels, hidden_channels), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_channels, num_classes)
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits, combining B-scans only after feature extraction."""
        expected_dimensions = 5 if self.slice_wise else self.dimensions + 2
        if images.ndim != expected_dimensions or images.shape[1] != self.in_channels:
            raise ValueError(f"Expected {expected_dimensions}D input with {self.in_channels} stored channels")
        if self.use_coordinate_channels:
            images = self.append_coordinate_channels(images)
        if self.slice_wise:
            batch_size, channels, depth, height, width = images.shape
            images = images.permute(0, 2, 1, 3, 4).reshape(batch_size * depth, channels, height, width)
        features = self.stages(images)
        spatial_axes = tuple(range(2, features.ndim))
        pooled_features = features.mean(dim=spatial_axes)
        if self.pooling == "mean_max":
            pooled_features = torch.cat((pooled_features, features.amax(dim=spatial_axes)), dim=1)
        if self.slice_wise:
            slice_features = pooled_features.reshape(batch_size, depth, -1)
            pooled_features = slice_features.mean(dim=1)
            if self.slice_pooling == "mean_max":
                pooled_features = torch.cat((pooled_features, slice_features.amax(dim=1)), dim=1)
        return self.classifier(pooled_features)
