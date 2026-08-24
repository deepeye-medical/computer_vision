"""Classification training, scheduling, and configuration checkpoints."""

from collections.abc import Mapping
from typing import Any, cast

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from pydantic import Field, model_validator
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassPrecision, MulticlassRecall, MulticlassSpecificity

from computer_vision.config import ConfigModel
from computer_vision.loss import CrossEntropyConfig, LossConfig
from computer_vision.model import VisionNetwork


class ConfigCheckpoint(pl.Callback):
    """Store resolved configuration with model weights."""

    def __init__(self, config: Mapping[str, object]) -> None:
        """Store the resolved configuration.

        Parameters
        ----------
        config
            Serializable settings used to reconstruct the run.
        """
        super().__init__()
        self.config = dict(config)

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict[str, Any]) -> None:  # noqa: ARG002
        """Attach settings to each checkpoint."""
        checkpoint["run_config"] = self.config






class TrainingConfig(ConfigModel):
    """Classification loss and AdamW settings."""

    learning_rate: float = Field(default=1e-3, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=1e-4, ge=0, allow_inf_nan=False)
    loss: LossConfig = Field(default_factory=CrossEntropyConfig)



class ClassificationModule(pl.LightningModule):
    """Train a classifier and report classification metrics."""

    def __init__(self, config: TrainingConfig, network: VisionNetwork) -> None:
        """Store the network and optimizer settings.

        Parameters
        ----------
        config
            Loss and optimizer settings.
        network
            Network that returns class logits.
        """
        super().__init__()
        self.config = config
        self.network = network

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits."""
        return self.network(images)

    def shared_step(self, batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        """Compute classification loss and sample-weighted epoch metrics."""
        images, labels = batch[:2]
        logits = self(images)
        loss = F.cross_entropy(logits, labels, label_smoothing=self.config.loss.label_smoothing)
        self.log(f"accuracy/{stage}", (logits.argmax(dim=1) == labels).float().mean(), on_step=False, on_epoch=True, batch_size=len(labels))
        self.log(f"loss/{stage}", loss, on_step=False, on_epoch=True, batch_size=len(labels), sync_dist=True)
        return loss

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Compute the optimization loss."""
        return self.shared_step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:  # noqa: ARG002
        """Report validation metrics."""
        self.shared_step(batch, "validation")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:  # noqa: ARG002
        """Report held-out test metrics."""
        self.shared_step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer | OptimizerLRSchedulerConfig:
        """Build AdamW with an optional warmup and cosine schedule."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        return optimizer
