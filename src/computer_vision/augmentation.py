"""Typed image augmentations shared by training and preview tools."""

from typing import Annotated, Self

import kornia.augmentation as K
import torch
from pydantic import Field, model_validator
from torch import nn

from computer_vision.config import ConfigModel

Probability = Annotated[float, Field(ge=0, le=1)]
Positive = Annotated[float, Field(gt=0)]


class SpatialAugmentationConfig(ConfigModel):
    """Spatial changes applied independently to each training image."""

    horizontal_flip_probability: Probability = 0.5
    """Probability of reflecting the horizontal image axis."""
    vertical_flip_probability: Probability = 0.0
    """Probability of reflecting the vertical image axis."""
    affine_probability: Probability = 0.0
    """Probability of applying the configured affine transform."""
    rotation_degrees: float = Field(default=0.0, ge=0, le=180)
    """Maximum absolute rotation angle."""
    max_translation: tuple[Probability, Probability] = (0.05, 0.05)
    """Maximum horizontal and vertical translation as image-size fractions."""
    shear_degrees: float = Field(default=0.0, ge=0, lt=90)
    """Maximum absolute horizontal shear angle."""


class IntensityAugmentationConfig(ConfigModel):
    """Intensity changes for RGB inputs scaled to [0, 1]."""

    brightness_probability: Probability = 0.0
    """Probability of an additive brightness shift."""
    contrast_probability: Probability = 0.0
    """Probability of contrast scaling."""
    gamma_probability: Probability = 0.0
    """Probability of a gamma adjustment."""
    noise_probability: Probability = 0.0
    """Probability of independent Gaussian pixel noise."""
    max_brightness_shift: Probability = 0.03
    """Maximum absolute brightness offset for inputs in [0, 1]."""
    contrast_range: tuple[Positive, Positive] = (0.9, 1.1)
    """Lower and upper contrast factors."""
    gamma_range: tuple[Positive, Positive] = (0.9, 1.1)
    """Lower and upper gamma exponents."""
    noise_std: float = Field(default=0.01, ge=0)
    """Noise standard deviation for inputs in [0, 1]."""

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        """Reject reversed bounds before building transforms."""
        if self.contrast_range[0] > self.contrast_range[1] or self.gamma_range[0] > self.gamma_range[1]:
            raise ValueError("Contrast and gamma ranges must have ordered bounds")
        return self


class AugmentationConfig(ConfigModel):
    """Separate spatial and intensity settings for training."""

    spatial: SpatialAugmentationConfig = SpatialAugmentationConfig()
    intensity: IntensityAugmentationConfig = IntensityAugmentationConfig()


class ImageAugmentation(nn.Module):
    """Apply the same batched image transforms in training and viewers."""

    def __init__(self, config: AugmentationConfig) -> None:
        """Build transforms from the shared configuration.

        Parameters
        ----------
        config
            Spatial and intensity settings for RGB images.
        """
        super().__init__()
        spatial = config.spatial
        intensity = config.intensity
        shift = intensity.max_brightness_shift
        self.transforms = nn.Sequential(
            K.RandomHorizontalFlip(p=spatial.horizontal_flip_probability, same_on_batch=False),
            K.RandomVerticalFlip(p=spatial.vertical_flip_probability, same_on_batch=False),
            K.RandomAffine(
                degrees=spatial.rotation_degrees,
                translate=spatial.max_translation,
                shear=(-spatial.shear_degrees, spatial.shear_degrees),
                p=spatial.affine_probability,
                same_on_batch=False,
            ),
            K.RandomBrightness(brightness=(1 - shift, 1 + shift), p=intensity.brightness_probability),
            K.RandomContrast(contrast=intensity.contrast_range, p=intensity.contrast_probability),
            K.RandomGamma(gamma=intensity.gamma_range, p=intensity.gamma_probability),
            K.RandomGaussianNoise(mean=0.0, std=intensity.noise_std, p=intensity.noise_probability),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Augment a batch of RGB images and keep its input range."""
        return self.transforms(images).clamp(0, 1)
