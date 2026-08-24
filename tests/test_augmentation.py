"""Check training-only augmentation and the shared input contract."""

from types import SimpleNamespace

import pytest
import torch

from computer_vision.augmentation import AugmentationConfig, IntensityAugmentationConfig, SpatialAugmentationConfig
from computer_vision.data import DatasetConfig, ImageDataModule


@pytest.mark.parametrize("training", [False, True])
def test_batch_augmentation(training: bool) -> None:
    """Flip only training images and preserve labels and stable sample indices."""
    config = DatasetConfig(augmentation=AugmentationConfig(spatial=SpatialAugmentationConfig(horizontal_flip_probability=1)))
    module = ImageDataModule(config)
    module.trainer = SimpleNamespace(training=training)  # ty: ignore[invalid-assignment]
    images = torch.arange(2 * 3 * 32 * 32).reshape(2, 3, 32, 32).float() / (2 * 3 * 32 * 32)
    labels = torch.tensor([0, 1])
    indices = torch.tensor([3, 5])
    augmented, returned_labels, returned_indices = module.on_after_batch_transfer((images, labels, indices), 0)
    torch.testing.assert_close(augmented, images.flip(-1) if training else images)
    assert returned_labels is labels and returned_indices is indices


def test_full_augmentation() -> None:
    """Exercise spatial and intensity transforms with a repeatable preview seed."""
    config = DatasetConfig(
        augmentation=AugmentationConfig(
            spatial=SpatialAugmentationConfig(affine_probability=1, rotation_degrees=10, shear_degrees=5),
            intensity=IntensityAugmentationConfig(brightness_probability=1, contrast_probability=1, gamma_probability=1, noise_probability=1),
        )
    )
    module = ImageDataModule(config)
    images = torch.linspace(0, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        first = module.augmentation(images)
        torch.manual_seed(42)
        second = module.augmentation(images)
    torch.testing.assert_close(first, second)
    assert first.shape == images.shape
    assert torch.isfinite(first).all() and 0 <= first.min() <= first.max() <= 1
    assert not torch.equal(first, images)
