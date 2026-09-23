"""Vision transformer for images, independent B-scans, and volumes."""

import logging
import math
from typing import Annotated, Literal

import torch
import torch.nn.functional as F
from jsonargparse.typing import register_type
from pydantic import Field, model_validator
from torch import nn
from torchvision import models
from torchvision.models.vision_transformer import Encoder

from computer_vision.config import ConfigModel
from computer_vision.model.base import Dropout, VisionNetwork

SpatialSize = tuple[Annotated[int, Field(gt=0)], ...]
TokenPooling = Literal["cls", "mean", "mean_max"]
SlicePooling = Literal["mean", "mean_max"]
Weights = Literal["vit_b_16_imagenet1k_v1"]
WEIGHTS = {"vit_b_16_imagenet1k_v1": models.ViT_B_16_Weights.IMAGENET1K_V1}
LOGGER = logging.getLogger(__name__)


class PatchConfig(ConfigModel):
    """Configure a fixed input grid and non-overlapping image or volume patches."""

    input_size: SpatialSize = (224, 224)
    """Input height and width, or depth, height, and width for volumes."""
    patch_size: SpatialSize = (16, 16)
    """Patch kernel and stride along each spatial axis."""

    @model_validator(mode="after")
    def validate_grid(self) -> "PatchConfig":
        """Prevent mismatched axes and silent truncation of partial patches."""
        if len(self.input_size) not in (2, 3) or len(self.patch_size) != len(self.input_size):
            raise ValueError("input_size and patch_size must contain two or three matching axes")
        if any(size % patch != 0 for size, patch in zip(self.input_size, self.patch_size, strict=True)):
            raise ValueError("Each input_size axis must be divisible by its patch_size")
        return self


class EncoderConfig(ConfigModel):
    """Configure the repeated transformer encoder blocks."""

    num_layers: int = Field(default=12, gt=0)
    """Number of transformer blocks."""
    num_heads: int = Field(default=12, gt=0)
    """Number of attention heads in each block."""
    hidden_dim: int = Field(default=768, gt=0)
    """Number of features per token."""
    mlp_dim: int = Field(default=3072, gt=0)
    """Number of hidden features in each block's MLP."""
    dropout: Dropout = 0.0
    """Dropout within the transformer encoder."""
    attention_dropout: Dropout = 0.0
    """Dropout on attention weights."""

    @model_validator(mode="after")
    def validate_heads(self) -> "EncoderConfig":
        """Require equally sized attention heads."""
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        return self


for config_type in (PatchConfig, EncoderConfig):
    register_type(config_type, serializer=config_type.model_dump, deserializer=config_type.model_validate)

DEFAULT_PATCHES = PatchConfig()
DEFAULT_ENCODER = EncoderConfig()


class PatchEmbedding(nn.Module):
    """Project non-overlapping image or volume patches into token vectors."""

    def __init__(self, in_channels: int, hidden_dim: int, config: PatchConfig) -> None:
        """Build the strided patch projection without pooling or padding.

        Parameters
        ----------
        in_channels
            Number of input channels, including optional coordinates.
        hidden_dim
            Number of features per token.
        config
            Fixed input size and patch dimensions.
        """
        super().__init__()
        if len(config.patch_size) == 2:
            kernel = (config.patch_size[0], config.patch_size[1])
            self.projection: nn.Conv2d | nn.Conv3d = nn.Conv2d(in_channels, hidden_dim, kernel, stride=kernel)
        else:
            kernel_3d = (config.patch_size[0], config.patch_size[1], config.patch_size[2])
            self.projection = nn.Conv3d(in_channels, hidden_dim, kernel_3d, stride=kernel_3d)
        nn.init.trunc_normal_(self.projection.weight, std=math.sqrt(1 / (in_channels * math.prod(config.patch_size))))
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return tokens shaped as (batch, patches, features)."""
        return self.projection(images).flatten(2).transpose(1, 2)


class ViT(VisionNetwork):
    """Classify images, independent B-scans, or volumes with a transformer."""

    input_mean: torch.Tensor
    input_std: torch.Tensor

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        patches: PatchConfig = DEFAULT_PATCHES,
        encoder: EncoderConfig = DEFAULT_ENCODER,
        weights: Weights | None = None,
        slice_wise: bool = False,
        pooling: TokenPooling = "cls",
        slice_pooling: SlicePooling = "mean",
        dropout: Dropout = 0.2,
        use_coordinate_channels: bool = False,
    ) -> None:
        """Build the patch projection, encoder, and task head without downloads.

        Parameters
        ----------
        in_channels
            Number of stored image channels, excluding coordinates.
        num_classes
            Number of task classes.
        patches
            Input grid and patch kernel sizes. Three axes select volume mode.
        encoder
            Transformer depth, token width, attention heads, and dropout.
        weights
            Pretrained source and normalization. Training loads the weights;
            direct callers must call ``load_pretrained_weights``.
        slice_wise
            Encode each B-scan independently with shared 2D patches and attention.
        pooling
            Use the class token or pool encoded patch tokens spatially.
        slice_pooling
            Combine independent B-scan features before the task head.
        dropout
            Dropout before the task classifier.
        use_coordinate_channels
            Append signed tensor coordinates after augmentation. Slice inputs
            retain each B-scan's original depth position.
        """
        super().__init__(in_channels)
        dimensions = len(patches.input_size)
        if num_classes < 2:
            raise ValueError("Require at least two classes")
        if slice_wise and dimensions != 2:
            raise ValueError("slice_wise requires 2D patches")
        self.num_classes = num_classes
        self.input_size = patches.input_size
        self.patch_grid = tuple(size // patch for size, patch in zip(patches.input_size, patches.patch_size, strict=True))
        self.slice_wise = slice_wise
        self.pooling = pooling
        self.slice_pooling = slice_pooling
        self.weights = weights
        self.use_coordinate_channels = use_coordinate_channels
        self.image_size = patches.input_size[-1] if patches.input_size[-2] == patches.input_size[-1] else None
        coordinate_channels = (3 if slice_wise else dimensions) if use_coordinate_channels else 0
        self.patch_embedding = PatchEmbedding(in_channels + coordinate_channels, encoder.hidden_dim, patches)
        self.class_token = nn.Parameter(torch.zeros(1, 1, encoder.hidden_dim))
        self.encoder = Encoder(
            seq_length=math.prod(self.patch_grid) + 1,
            num_layers=encoder.num_layers,
            num_heads=encoder.num_heads,
            hidden_dim=encoder.hidden_dim,
            mlp_dim=encoder.mlp_dim,
            dropout=encoder.dropout,
            attention_dropout=encoder.attention_dropout,
        )
        classifier_channels = encoder.hidden_dim * (2 if pooling == "mean_max" else 1) * (2 if slice_wise and slice_pooling == "mean_max" else 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(classifier_channels, num_classes))
        self.register_buffer("input_mean", torch.zeros(in_channels))
        self.register_buffer("input_std", torch.ones(in_channels))
        if weights is not None:
            preprocessing = WEIGHTS[weights].transforms()
            mean, std = torch.tensor(preprocessing.mean), torch.tensor(preprocessing.std)
            self.input_mean.copy_(mean if in_channels == 3 else mean.mean().repeat(in_channels))
            self.input_std.copy_(std if in_channels == 3 else std.mean().repeat(in_channels))

    def load_pretrained_weights(self) -> None:
        """Load the encoder and adapt image patches and positions to the input grid.

        Non-RGB channels receive the summed RGB filter divided by their count.
        Coordinate filters start at zero. Volume patch filters repeat uniformly
        across depth and divide by the depth patch size. Position embeddings
        resize spatially and repeat across depth without scaling; each volume
        position then learns independently. The task head stays random.
        """
        if self.weights is None:
            return
        source = WEIGHTS[self.weights].get_state_dict(progress=True, check_hash=True)
        target = self.state_dict()
        transferred: dict[str, torch.Tensor] = {}
        used: set[str] = set()
        for name, target_weight in target.items():
            if name.startswith(("classifier.", "input_")):
                continue
            source_name = name.replace("patch_embedding.projection.", "conv_proj.")
            if source_name not in source:
                # ImageNet V1 checkpoints can use the original torchvision MLP names.
                source_name = source_name.replace(".mlp.0.", ".mlp.linear_1.").replace(".mlp.3.", ".mlp.linear_2.")
            if source_name not in source:
                raise ValueError(f"Pretrained checkpoint has no parameter for {name}")
            source_weight = source[source_name]
            if source_name == "conv_proj.weight":
                if self.in_channels != 3:
                    source_weight = source_weight.sum(dim=1, keepdim=True).repeat(1, self.in_channels, 1, 1) / self.in_channels
                if self.use_coordinate_channels:
                    coordinate_shape = (source_weight.shape[0], target_weight.shape[1] - self.in_channels, *source_weight.shape[2:])
                    source_weight = torch.cat((source_weight, source_weight.new_zeros(coordinate_shape)), dim=1)
                if target_weight.ndim == 5:
                    depth = target_weight.shape[2]
                    source_weight = source_weight.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth
            elif source_name == "encoder.pos_embedding":
                spatial_count = source_weight.shape[1] - 1
                source_size = math.isqrt(spatial_count)
                if source_size**2 != spatial_count:
                    raise ValueError("Pretrained position embeddings must use a square image grid")
                spatial_positions = source_weight[:, 1:].transpose(1, 2).reshape(1, -1, source_size, source_size)
                if self.patch_grid[-2:] != (source_size, source_size):
                    spatial_positions = F.interpolate(spatial_positions, size=self.patch_grid[-2:], mode="bicubic", align_corners=True)
                spatial_positions = spatial_positions.flatten(2).transpose(1, 2)
                if len(self.patch_grid) == 3:
                    spatial_positions = spatial_positions.repeat(1, self.patch_grid[0], 1)
                source_weight = torch.cat((source_weight[:, :1], spatial_positions), dim=1)
            if source_weight.shape != target_weight.shape:
                raise ValueError(f"Pretrained shape mismatch for {name}: {tuple(source_weight.shape)} != {tuple(target_weight.shape)}")
            transferred[name] = source_weight
            used.add(source_name)
        unused = source.keys() - used - {"heads.head.weight", "heads.head.bias"}
        if unused:
            raise ValueError(f"Network does not use pretrained parameters: {sorted(unused)[:5]}")
        self.load_state_dict(target | transferred)
        LOGGER.info("Loaded %s encoder weights", self.weights)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return task logits from a fixed grid, joining slices only at the head."""
        dimensions = len(self.input_size)
        expected_dimensions = 5 if self.slice_wise else dimensions + 2
        expected_spatial_size = images.shape[-2:] if self.slice_wise else images.shape[2:]
        if images.ndim != expected_dimensions or images.shape[1] != self.in_channels or expected_spatial_size != self.input_size:
            raise ValueError(f"Expected {expected_dimensions}D input with {self.in_channels} stored channels and spatial size {self.input_size}")
        normalization_shape = (1, self.in_channels, *([1] * (images.ndim - 2)))
        images = (images - self.input_mean.view(normalization_shape)) / self.input_std.view(normalization_shape)
        if self.use_coordinate_channels:
            images = self.append_coordinate_channels(images)
        if self.slice_wise:
            batch_size, channels, depth, height, width = images.shape
            images = images.permute(0, 2, 1, 3, 4).reshape(batch_size * depth, channels, height, width)
        tokens = self.patch_embedding(images)
        class_tokens = self.class_token.expand(tokens.shape[0], -1, -1)
        encoded = self.encoder(torch.cat((class_tokens, tokens), dim=1))
        if self.pooling == "cls":
            pooled_features = encoded[:, 0]
        else:
            patch_features = encoded[:, 1:]
            pooled_features = patch_features.mean(dim=1)
            if self.pooling == "mean_max":
                pooled_features = torch.cat((pooled_features, patch_features.amax(dim=1)), dim=1)
        if self.slice_wise:
            slice_features = pooled_features.reshape(batch_size, depth, -1)
            pooled_features = slice_features.mean(dim=1)
            if self.slice_pooling == "mean_max":
                pooled_features = torch.cat((pooled_features, slice_features.amax(dim=1)), dim=1)
        return self.classifier(pooled_features)
