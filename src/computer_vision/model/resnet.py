"""Stage-configured residual networks for images, slices, and volumes."""

import logging
from collections.abc import Callable
from typing import Annotated, Literal, cast

import torch
from jsonargparse.typing import register_type
from pydantic import Field, field_validator, model_validator
from torch import nn
from torchvision import models

from computer_vision.config import ConfigModel
from computer_vision.model.base import Dropout, VisionNetwork

SpatialSize = tuple[Annotated[int, Field(gt=0)], ...]
BlockType = Literal["standard", "factorized"]
Pooling = Literal["mean", "mean_max"]
Weights = Literal["resnet18_imagenet1k_v1", "resnet34_imagenet1k_v1", "r2plus1d_18_kinetics400_v1"]
PRETRAINED_WEIGHTS = {
    "resnet18_imagenet1k_v1": models.ResNet18_Weights.IMAGENET1K_V1,
    "resnet34_imagenet1k_v1": models.ResNet34_Weights.IMAGENET1K_V1,
    "r2plus1d_18_kinetics400_v1": models.video.R2Plus1D_18_Weights.KINETICS400_V1,
}
LOGGER = logging.getLogger(__name__)


class StemConfig(ConfigModel):
    """Configure feature extraction before the residual stages."""

    out_channels: int = Field(default=16, gt=0)
    """Number of stem output channels."""
    kernel_size: SpatialSize = (3, 3)
    """Odd kernel sizes in spatial tensor order."""
    stride: SpatialSize = (1, 1)
    """Convolution strides in spatial tensor order."""
    pool_size: SpatialSize | None = None
    """Optional max-pooling kernel sizes."""
    pool_stride: SpatialSize | None = None
    """Max-pooling strides; required when pooling is enabled."""
    intermediate_channels: int | None = Field(default=None, gt=0)
    """Spatial convolution output channels for a factorized stem."""

    @field_validator("kernel_size")
    @classmethod
    def validate_kernel(cls, kernel_size: SpatialSize) -> SpatialSize:
        """Require odd image or volume kernels for symmetric padding."""
        if len(kernel_size) not in (2, 3) or any(size % 2 == 0 for size in kernel_size):
            raise ValueError("kernel_size must contain two or three positive odd values")
        return kernel_size

    @model_validator(mode="after")
    def validate_dimensions(self) -> "StemConfig":
        """Keep convolution and pooling axes consistent."""
        if (self.pool_size is None) != (self.pool_stride is None):
            raise ValueError("pool_size and pool_stride must be supplied together")
        sizes = (self.stride, self.pool_size, self.pool_stride)
        if any(size is not None and len(size) != len(self.kernel_size) for size in sizes):
            raise ValueError("Stem kernels, strides, and pooling must have the same dimensionality")
        if self.pool_size is not None and any(size % 2 == 0 for size in self.pool_size):
            raise ValueError("pool_size values must be odd for symmetric padding")
        return self


class ResStageConfig(ConfigModel):
    """Configure repeated residual blocks and the first block's stride."""

    out_channels: int = Field(gt=0)
    """Number of output channels in each residual block."""
    num_blocks: int = Field(default=1, gt=0)
    """Number of residual blocks in the stage."""
    kernel_size: SpatialSize = (3, 3)
    """Odd convolution kernel sizes in spatial tensor order."""
    stride: SpatialSize = (1, 1)
    """First block strides; remaining blocks preserve spatial dimensions."""

    @field_validator("kernel_size")
    @classmethod
    def validate_kernel(cls, kernel_size: SpatialSize) -> SpatialSize:
        """Require odd image or volume kernels for symmetric padding."""
        return StemConfig.validate_kernel(kernel_size)

    @model_validator(mode="after")
    def validate_dimensions(self) -> "ResStageConfig":
        """Keep convolution and stride axes consistent."""
        if len(self.stride) != len(self.kernel_size):
            raise ValueError("Stage kernels and strides must have the same dimensionality")
        return self


for config_type in (StemConfig, ResStageConfig):
    register_type(config_type, serializer=config_type.model_dump, deserializer=config_type.model_validate)


class ResBlock(nn.Module):
    """Standard two-convolution residual block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, ...] = (3, 3), stride: tuple[int, ...] = (1, 1)) -> None:
        """Build a block with matching strides in the residual and skip branches.

        Parameters
        ----------
        in_channels
            Number of input feature channels.
        out_channels
            Number of output feature channels.
        kernel_size
            Odd convolution kernel sizes in spatial tensor order.
        stride
            Strides for the first convolution and skip projection.
        """
        super().__init__()
        convolution = cast(Callable[..., nn.Module], nn.Conv2d if len(kernel_size) == 2 else nn.Conv3d)
        normalization = nn.BatchNorm2d if len(kernel_size) == 2 else nn.BatchNorm3d
        padding = tuple(size // 2 for size in kernel_size)
        self.conv1 = convolution(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = normalization(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convolution(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = normalization(out_channels)
        self.downsample = (
            nn.Identity()
            if in_channels == out_channels and all(size == 1 for size in stride)
            else nn.Sequential(convolution(in_channels, out_channels, 1, stride=stride, bias=False), normalization(out_channels))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Combine learned features with the projected input."""
        residual = self.downsample(features)
        features = self.relu(self.bn1(self.conv1(features)))
        return self.relu(self.bn2(self.conv2(features)) + residual)


class FactorizedConv(nn.Sequential):
    """Spatial convolution followed by a convolution across slices."""

    def __init__(
        self, in_channels: int, out_channels: int, intermediate_channels: int, kernel_size: tuple[int, ...], stride: tuple[int, ...]
    ) -> None:
        """Separate spatial and depth filtering while preserving channel mixing.

        Parameters
        ----------
        in_channels
            Number of input feature channels.
        out_channels
            Number of output feature channels.
        intermediate_channels
            Number of channels between spatial and depth convolutions.
        kernel_size
            Depth, height, and width kernel sizes.
        stride
            Depth, height, and width strides.
        """
        depth, height, width = kernel_size
        super().__init__(
            nn.Conv3d(in_channels, intermediate_channels, (1, height, width), (1, stride[1], stride[2]), (0, height // 2, width // 2), bias=False),
            nn.BatchNorm3d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(intermediate_channels, out_channels, (depth, 1, 1), (stride[0], 1, 1), (depth // 2, 0, 0), bias=False),
        )


class FactorizedResBlock(nn.Module):
    """R(2+1)D residual block with separate spatial and depth filtering."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, ...], stride: tuple[int, ...]) -> None:
        """Build a factorized block using the torchvision channel rule.

        Parameters
        ----------
        in_channels
            Number of input feature channels.
        out_channels
            Number of output feature channels.
        kernel_size
            Odd kernel sizes in depth, height, and width order.
        stride
            Strides for the first convolution pair and skip projection.
        """
        super().__init__()
        depth, height, width = kernel_size
        intermediate_channels = max(1, in_channels * out_channels * depth * height * width // (in_channels * height * width + depth * out_channels))
        self.conv1 = nn.Sequential(
            FactorizedConv(in_channels, out_channels, intermediate_channels, kernel_size, stride),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Torchvision computes the intermediate channel count once per block.
        self.conv2 = nn.Sequential(
            FactorizedConv(out_channels, out_channels, intermediate_channels, kernel_size, (1, 1, 1)), nn.BatchNorm3d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Identity()
            if in_channels == out_channels and all(size == 1 for size in stride)
            else nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=(stride[0], stride[1], stride[2]), bias=False), nn.BatchNorm3d(out_channels)
            )
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Combine spatial and depth features with the projected input."""
        return self.relu(self.conv2(self.conv1(features)) + self.downsample(features))


class ResStage(nn.Sequential):
    """Repeated residual blocks with optional downsampling in the first block."""

    def __init__(self, in_channels: int, config: ResStageConfig, block_type: BlockType = "standard") -> None:
        """Build one residual stage.

        Parameters
        ----------
        in_channels
            Number of channels entering the stage.
        config
            Output channels, block count, kernels, and first-block strides.
        block_type
            Standard convolutions or separate spatial and depth convolutions.
        """
        block_class = ResBlock if block_type == "standard" else FactorizedResBlock
        super().__init__(
            *(
                block_class(
                    in_channels if index == 0 else config.out_channels,
                    config.out_channels,
                    config.kernel_size,
                    config.stride if index == 0 else (1,) * len(config.kernel_size),
                )
                for index in range(config.num_blocks)
            )
        )


class ResStem(nn.Sequential):
    """Standard or factorized input convolutions with optional max pooling."""

    def __init__(self, in_channels: int, config: StemConfig, block_type: BlockType = "standard") -> None:
        """Build the stem with layer indices matching torchvision checkpoints.

        Parameters
        ----------
        in_channels
            Number of input channels, including optional coordinates.
        config
            Output channels, convolution kernels, strides, and pooling.
        block_type
            Standard convolutions or separate spatial and depth convolutions.
        """
        dimensions = len(config.kernel_size)
        normalization = nn.BatchNorm2d if dimensions == 2 else nn.BatchNorm3d
        # Layer indices must match the source checkpoint's stem parameter names.
        if block_type == "standard":
            if config.intermediate_channels is not None:
                raise ValueError("stem.intermediate_channels is only used by factorized blocks")
            convolution = cast(Callable[..., nn.Module], nn.Conv2d if dimensions == 2 else nn.Conv3d)
            layers: list[nn.Module] = [
                convolution(
                    in_channels,
                    config.out_channels,
                    config.kernel_size,
                    stride=config.stride,
                    padding=tuple(size // 2 for size in config.kernel_size),
                    bias=False,
                ),
                normalization(config.out_channels),
                nn.ReLU(inplace=True),
            ]
        else:
            intermediate_channels = config.intermediate_channels
            if dimensions != 3 or intermediate_channels is None:
                raise ValueError("Factorized blocks require 3D kernels and stem.intermediate_channels")
            factorized_stem = FactorizedConv(in_channels, config.out_channels, intermediate_channels, config.kernel_size, config.stride)
            layers = [*factorized_stem.children(), nn.BatchNorm3d(config.out_channels), nn.ReLU(inplace=True)]
        if config.pool_size is not None:
            pooling_class = nn.MaxPool2d if dimensions == 2 else nn.MaxPool3d
            layers.append(pooling_class(config.pool_size, stride=config.pool_stride, padding=tuple(size // 2 for size in config.pool_size)))
        super().__init__(*layers)


DEFAULT_STEM = StemConfig()


class ResNet(VisionNetwork):
    """Classify images, independent B-scans, or volumes with configured stages."""

    input_mean: torch.Tensor
    input_std: torch.Tensor

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        stem: StemConfig = DEFAULT_STEM,
        stages: tuple[ResStageConfig, ...] = (
            ResStageConfig(out_channels=16),
            ResStageConfig(out_channels=32, stride=(2, 2)),
            ResStageConfig(out_channels=64, stride=(2, 2)),
            ResStageConfig(out_channels=96, stride=(2, 2)),
            ResStageConfig(out_channels=128, stride=(2, 2)),
        ),
        block_type: BlockType = "standard",
        weights: Weights | None = None,
        slice_wise: bool = False,
        pooling: Pooling = "mean_max",
        slice_pooling: Pooling = "mean",
        dropout: Dropout = 0.2,
        use_coordinate_channels: bool = False,
    ) -> None:
        """Build the network without downloading weights.

        Parameters
        ----------
        in_channels
            Number of stored image channels, excluding coordinates.
        num_classes
            Number of task classes.
        stem
            Convolution and optional pooling before the stages.
        stages
            Residual stages in execution order.
        block_type
            Standard convolutions or separate spatial and depth convolutions.
        weights
            Pretrained source and normalization. Training loads the weights;
            direct callers must call ``load_pretrained_weights``.
        slice_wise
            Process each B-scan independently with a shared 2D backbone.
        pooling
            Spatial feature pooling before the classifier or slice aggregation.
        slice_pooling
            Feature pooling across B-scans, used only for slice-wise inputs.
        dropout
            Dropout probability before the task classifier.
        use_coordinate_channels
            Append signed tensor coordinates after augmentation. Volume and
            slice-wise inputs include the original B-scan depth coordinate.
        """
        super().__init__(in_channels)
        dimensions = len(stem.kernel_size)
        if num_classes < 2 or not stages:
            raise ValueError("Require at least two classes and one residual stage")
        if any(len(stage.kernel_size) != dimensions for stage in stages):
            raise ValueError("Stem and stages must have the same dimensionality")
        if slice_wise and dimensions != 2:
            raise ValueError("slice_wise requires 2D kernels")
        self.dimensions = dimensions
        self.num_classes = num_classes
        self.slice_wise = slice_wise
        self.pooling = pooling
        self.slice_pooling = slice_pooling
        self.use_coordinate_channels = use_coordinate_channels
        self.weights = weights
        self.block_type = block_type
        coordinate_channels = (3 if slice_wise else dimensions) if use_coordinate_channels else 0
        stem_channels = in_channels + coordinate_channels
        self.stem = ResStem(stem_channels, stem, block_type)
        stage_layers: list[nn.Module] = []
        previous_channels = stem.out_channels
        for stage in stages:
            stage_layers.append(ResStage(previous_channels, stage, block_type))
            previous_channels = stage.out_channels
        self.stages = nn.Sequential(*stage_layers)
        classifier_channels = previous_channels * (2 if pooling == "mean_max" else 1) * (2 if slice_wise and slice_pooling == "mean_max" else 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(classifier_channels, num_classes))
        self.register_buffer("input_mean", torch.zeros(in_channels))
        self.register_buffer("input_std", torch.ones(in_channels))
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if weights is not None:
            expected_counts = (3, 4, 6, 3) if weights.startswith("resnet34") else (2, 2, 2, 2)
            expected_type = "factorized" if weights.startswith("r2plus1d") else "standard"
            if block_type != expected_type or tuple(stage.num_blocks for stage in stages) != expected_counts:
                raise ValueError("Configured block type and stage counts do not match the selected weights")
            if stem.out_channels != 64 or tuple(stage.out_channels for stage in stages) != (64, 128, 256, 512):
                raise ValueError("Pretrained weights require stem width 64 and stage widths 64, 128, 256, 512")
            if stem.kernel_size[-2:] != (7, 7) or any(stage.kernel_size[-2:] != (3, 3) for stage in stages):
                raise ValueError("Pretrained weights require a 7x7 stem and 3x3 spatial block kernels")
            if block_type == "factorized" and (
                stem.kernel_size != (3, 7, 7) or stem.intermediate_channels != 45 or any(stage.kernel_size != (3, 3, 3) for stage in stages)
            ):
                raise ValueError("R(2+1)D weights require depth kernels of size 3 and stem.intermediate_channels=45")
            preprocessing = PRETRAINED_WEIGHTS[weights].transforms()
            mean = torch.tensor(preprocessing.mean)
            std = torch.tensor(preprocessing.std)
            self.input_mean.copy_(mean if in_channels == 3 else mean.mean().repeat(in_channels))
            self.input_std.copy_(std if in_channels == 3 else std.mean().repeat(in_channels))

    def load_pretrained_weights(self) -> None:
        """Load the backbone, inflate 2D kernels, and leave the task head trainable.

        Non-RGB channels use the summed RGB filter divided by their count.
        Coordinate filters start at zero. Kernel inflation repeats spatial
        weights uniformly across depth and divides by the depth kernel size.
        The task head keeps its random initialization.
        """
        if self.weights is None:
            return
        source = PRETRAINED_WEIGHTS[self.weights].get_state_dict(progress=True, check_hash=True)
        target = self.state_dict()
        transferred: dict[str, torch.Tensor] = {}
        for name, target_weight in target.items():
            if name.startswith(("classifier.", "input_")):
                continue
            source_name = name
            if name.startswith("stages."):
                _, stage_index, block_name = name.split(".", 2)
                source_name = f"layer{int(stage_index) + 1}.{block_name}"
            elif self.block_type == "standard" and name.startswith("stem."):
                source_name = name.replace("stem.0.", "conv1.").replace("stem.1.", "bn1.")
            if source_name not in source:
                raise ValueError(f"Pretrained checkpoint has no parameter for {name}")
            source_weight = source[source_name]
            if name == "stem.0.weight":
                if self.in_channels != 3:
                    source_weight = source_weight.sum(dim=1, keepdim=True).repeat(1, self.in_channels, *([1] * (source_weight.ndim - 2)))
                    source_weight /= self.in_channels
                if self.use_coordinate_channels:
                    coordinate_shape = (source_weight.shape[0], target_weight.shape[1] - self.in_channels, *source_weight.shape[2:])
                    source_weight = torch.cat((source_weight, source_weight.new_zeros(coordinate_shape)), dim=1)
            if source_weight.ndim == 4 and target_weight.ndim == 5:
                depth = target_weight.shape[2]
                source_weight = source_weight.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth
            if source_weight.shape != target_weight.shape:
                raise ValueError(f"Pretrained shape mismatch for {name}: {tuple(source_weight.shape)} != {tuple(target_weight.shape)}")
            transferred[name] = source_weight
        self.load_state_dict(target | transferred)
        LOGGER.info("Loaded %s backbone weights", self.weights)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits from images or volumes; slice mode joins features."""
        expected_dimensions = 5 if self.slice_wise else self.dimensions + 2
        if images.ndim != expected_dimensions or images.shape[1] != self.in_channels:
            raise ValueError(f"Expected {expected_dimensions}D input with {self.in_channels} stored channels")
        normalization_shape = (1, self.in_channels, *([1] * (images.ndim - 2)))
        images = (images - self.input_mean.view(normalization_shape)) / self.input_std.view(normalization_shape)
        if self.use_coordinate_channels:
            images = self.append_coordinate_channels(images)
        if self.slice_wise:
            batch_size, channels, depth, height, width = images.shape
            images = images.permute(0, 2, 1, 3, 4).reshape(batch_size * depth, channels, height, width)
        features = self.stages(self.stem(images))
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
