"""RGB image loading with reproducible cross-validation folds."""

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import torch
from PIL import Image
from pydantic import Field, model_validator
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from computer_vision.augmentation import AugmentationConfig, ImageAugmentation
from computer_vision.config import ConfigModel
from computer_vision.manifest import read_manifest


class DatasetConfig(ConfigModel):
    """Image sources, transforms, and fold settings."""

    kind: Literal["fake", "imagefolder", "manifest"] = "fake"
    root: Path | None = None
    num_classes: int = Field(default=2, ge=2)
    image_size: int = Field(default=32, ge=16)
    num_samples: int = Field(default=60, ge=6)
    num_folds: int = Field(default=3, ge=2)
    validation_fold: int = Field(default=0, ge=0)
    split_seed: int = Field(default=42, ge=0, le=2**32 - 1)
    batch_size: int = Field(default=8, ge=1)
    num_workers: int = Field(default=0, ge=0)
    fold_strategy: Literal["group", "random"] = "group"
    """Manifest group folds or seeded image folds; other loaders use image folds."""

    augmentation: AugmentationConfig = AugmentationConfig()
    """Spatial and intensity transforms applied only to training batches."""

    @model_validator(mode="after")
    def validate_source(self) -> "DatasetConfig":
        """Reject missing image paths and invalid fold indices."""
        if self.kind in {"imagefolder", "manifest"} and self.root is None:
            raise ValueError("imagefolder and manifest require dataset.root")
        if self.validation_fold >= self.num_folds:
            raise ValueError("validation_fold must be smaller than num_folds")
        return self


def image_transform(config: DatasetConfig) -> transforms.Compose:
    """Share deterministic RGB preprocessing between training and the viewer."""
    return transforms.Compose([transforms.Resize((config.image_size, config.image_size)), transforms.ToTensor()])


def assign_group_folds(group_ids: list[str], num_folds: int) -> list[int]:
    """Rank groups by sample count, then assign folds without separating a group.

    Group ID breaks ties. Assignment ignores row order and training seeds.
    This spreads large groups across folds but does not stratify labels.

    Parameters
    ----------
    group_ids
        Group identity for each training-pool sample.
    num_folds
        Number of validation folds.

    Returns
    -------
    list[int]
        Fold index for each input sample, in input order.
    """
    counts = Counter(group_ids)
    if len(counts) < num_folds:
        raise ValueError("The group count must be at least num_folds")
    ranked_groups = sorted(counts, key=lambda group: (-counts[group], group))
    fold_by_group = {group: rank % num_folds for rank, group in enumerate(ranked_groups)}
    return [fold_by_group[group] for group in group_ids]


class ManifestDataset(Dataset):
    """Load the integrated manifest with consistent class indices."""

    def __init__(self, config: DatasetConfig, test: bool = False) -> None:
        """Select a split and validate its class names and count.

        Parameters
        ----------
        config
            Dataset settings; root is the manifest file path.
        test
            Select held-out test images instead of training images.
        """
        assert config.root is not None  # noqa: S101 - validated configuration boundary.
        self.root = config.root
        records = read_manifest(self.root)
        if len({record.sample_id for record in records}) != len(records):
            raise ValueError("Manifest sample IDs must be unique")
        groups_by_split = {split: {record.group_id or record.sample_id for record in records if record.split == split} for split in ("train", "test")}
        if groups_by_split["train"] & groups_by_split["test"]:
            raise ValueError("Manifest groups must not overlap train and test")
        classes = sorted({record.label for record in records})
        if any(not label.strip() for label in classes) or len(classes) != config.num_classes:
            raise ValueError("Manifest labels must be present and match dataset.num_classes")
        self.classes = classes
        self.class_to_idx = {label: index for index, label in enumerate(classes)}
        self.records = [record for record in records if record.split == ("test" if test else "train")]
        if not self.records:
            raise ValueError("Requested manifest split is empty")
        self.transform = image_transform(config)

    def __len__(self) -> int:
        """Return the number of images in this split."""
        return len(self.records)

    def original_image(self, index: int) -> Image.Image:
        """Read an RGB image using the same path resolution as training."""
        with Image.open(self.root.parent / self.records[index].path) as image:
            return image.convert("RGB")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return the model input and its class index."""
        record = self.records[index]
        return self.transform(self.original_image(index)), self.class_to_idx[record.label]


class IndexedSubset(Subset):
    """Attach a stable training-split index for prediction history."""

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        """Return an image, label, and stable local index."""
        image, label = self.dataset[self.indices[idx]]
        return image, label, idx

    def __getitems__(self, indices: list[int]) -> list[tuple[torch.Tensor, int, int]]:
        """Keep sample indices when the loader fetches a batch."""
        return [self[index] for index in indices]


class ImageDataModule(pl.LightningDataModule):
    """Keep fold membership independent of training seeds."""

    def __init__(self, config: DatasetConfig) -> None:
        """Store dataset settings.

        Parameters
        ----------
        config
            Image source, transforms, and loader settings.
        """
        super().__init__()
        self.config = config
        self.augmentation = ImageAugmentation(config.augmentation)
        self.train_dataset: IndexedSubset
        self.validation_dataset: Subset
        self.test_dataset: Dataset

    def make_dataset(self, test: bool = False) -> Dataset:
        """Build an image source with deterministic preprocessing."""
        config = self.config
        transform = image_transform(config)
        if config.kind == "fake":
            return datasets.FakeData(
                size=config.num_samples,
                image_size=(3, config.image_size, config.image_size),
                num_classes=config.num_classes,
                transform=transform,
                random_offset=config.num_samples if test else 0,
            )
        assert config.root is not None  # noqa: S101 - validated configuration boundary.
        if config.kind == "manifest":
            return ManifestDataset(config, test=test)
        images = datasets.ImageFolder(config.root / ("test" if test else "train"), transform=transform)
        if len(images.classes) != config.num_classes:
            raise ValueError("ImageFolder class count must match dataset.num_classes")
        if test and images.class_to_idx != datasets.ImageFolder(config.root / "train").class_to_idx:
            raise ValueError("Train and test class names must match")
        return images

    def setup(self, stage: str | None = None) -> None:
        """Split training images into fixed folds; load test images separately."""
        if stage in (None, "fit", "validate"):
            training = self.make_dataset()
            count = len(training)  # ty: ignore[invalid-argument-type]
            if count < self.config.num_folds:
                raise ValueError("The image count must be at least num_folds")
            fold = self.config.validation_fold
            if isinstance(training, ManifestDataset) and self.config.fold_strategy == "group":
                assignments = assign_group_folds([record.group_id or record.sample_id for record in training.records], self.config.num_folds)
                training_indices = [index for index, assignment in enumerate(assignments) if assignment != fold]
                validation_indices = [index for index, assignment in enumerate(assignments) if assignment == fold]
            else:
                indices = torch.randperm(count, generator=torch.Generator().manual_seed(self.config.split_seed)).tolist()
                training_indices = [index for position, index in enumerate(indices) if position % self.config.num_folds != fold]
                validation_indices = indices[fold :: self.config.num_folds]
            self.train_dataset = IndexedSubset(training, training_indices)
            self.validation_dataset = Subset(training, validation_indices)
            logging.getLogger(__name__).info("Split samples: training=%d, validation=%d", len(training_indices), len(validation_indices))
        if stage in (None, "test"):
            self.test_dataset = self.make_dataset(test=True)

    def training_fingerprint(self) -> str:
        """Identify ordered training samples, labels, and split settings."""
        source = self.train_dataset.dataset
        if isinstance(source, datasets.ImageFolder):
            samples = [
                (Path(source.samples[index][0]).relative_to(source.root).as_posix(), source.samples[index][1]) for index in self.train_dataset.indices
            ]
        elif isinstance(source, ManifestDataset):
            samples = [
                (source.records[index].sample_id, source.records[index].group_id, source.records[index].label, source.records[index].path)
                for index in self.train_dataset.indices
            ]
        else:
            samples = list(self.train_dataset.indices)
        identity = {"dataset": self.config.model_dump(mode="json", exclude={"root", "num_workers", "batch_size"}), "samples": samples}
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def train_dataloader(self) -> DataLoader:
        """Shuffle training images using the seeded Lightning workers."""
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Load validation images without shuffling."""
        return DataLoader(self.validation_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

    def test_dataloader(self) -> DataLoader:
        """Load held-out test images without shuffling."""
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

    def on_after_batch_transfer(self, batch: tuple[torch.Tensor, ...], dataloader_idx: int) -> tuple[torch.Tensor, ...]:
        """Augment training batches on their device; keep evaluation deterministic."""
        del dataloader_idx
        if self.trainer is not None and self.trainer.training:
            images, *metadata = batch
            return (self.augmentation.to(images.device)(images), *metadata)
        return tuple(batch)
