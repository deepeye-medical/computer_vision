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
from computer_vision.data import ImageDataModule
from computer_vision.loss import BCEConfig, CrossEntropyConfig, ELRRegularization, FocalConfig, FocalLoss, GeneralizedCrossEntropyLoss, LossConfig
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
        data_module = getattr(trainer, "datamodule", None)
        if isinstance(data_module, ImageDataModule):
            source = data_module.train_dataset.dataset
            classes = getattr(source, "classes", None)
            if classes is not None:
                checkpoint["class_names"] = list(classes)


class CosineScheduleConfig(ConfigModel):
    """Linear warmup followed by cosine decay."""

    warmup_epochs: int = Field(default=5, ge=1)
    start_factor: float = Field(default=0.1, gt=0, le=1)
    min_factor: float = Field(default=0.01, ge=0, le=1)


class ELRConfig(ConfigModel):
    """Early-learning regularization for binary cross entropy."""

    beta: float = Field(default=0.7, ge=0, lt=1)
    strength: float = Field(default=3.0, gt=0, allow_inf_nan=False)


class TrainingConfig(ConfigModel):
    """Classification loss and AdamW settings."""

    learning_rate: float = Field(default=1e-3, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=1e-4, ge=0, allow_inf_nan=False)
    loss: LossConfig = Field(default_factory=CrossEntropyConfig)
    scheduler: CosineScheduleConfig | None = None
    elr: ELRConfig | None = None

    @model_validator(mode="after")
    def validate_elr(self) -> "TrainingConfig":
        """Restrict ELR to the binary BCE objective."""
        if self.elr is not None and not isinstance(self.loss, BCEConfig):
            raise ValueError("ELR requires loss.name=bce")
        return self


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
        self.class_names: tuple[str, ...] | None = None
        num_classes = getattr(network, "num_classes", 2)
        metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
                "balanced_accuracy": MulticlassAccuracy(num_classes=num_classes, average="macro"),
                "auroc": MulticlassAUROC(num_classes=num_classes),
                "precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
                "recall": MulticlassRecall(num_classes=num_classes, average="macro"),
                "specificity": MulticlassSpecificity(num_classes=num_classes, average="macro"),
            }
        )
        self.metrics = nn.ModuleDict({f"split_{stage}": metrics.clone(postfix=f"/{stage}") for stage in ("train", "validation", "test")})
        self.elr: ELRRegularization | None = None
        self.pending_elr_state: dict[str, Any] | None = None
        self.elr_fingerprint: str | None = None

    def on_train_start(self) -> None:
        """Allocate prediction history for this run's fixed training split."""
        if self.config.elr is not None:
            if self.trainer.world_size != 1:
                raise ValueError("ELR requires a single device")
            data_module = getattr(self.trainer, "datamodule", None)
            if not isinstance(data_module, ImageDataModule):
                raise ValueError("ELR requires ImageDataModule")
            self.elr_fingerprint = data_module.training_fingerprint()
            self.elr = ELRRegularization(len(data_module.train_dataset), self.config.elr.beta).to(self.device)
            if self.pending_elr_state is not None:
                saved = self.pending_elr_state
                if saved["fingerprint"] != self.elr_fingerprint or saved["config"] != self.config.elr.model_dump():
                    raise ValueError("ELR resume requires matching training membership, labels, and settings")
                history = saved["prediction_history"]
                if not isinstance(history, torch.Tensor) or history.shape != self.elr.prediction_history.shape:
                    raise ValueError("ELR history shape does not match this training split")
                self.elr.prediction_history.copy_(history)
                self.pending_elr_state = None

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Keep prediction history and sample identity with the checkpoint."""
        if self.elr is not None and self.config.elr is not None:
            checkpoint["elr_state"] = {
                "prediction_history": self.elr.prediction_history.cpu().clone(),
                "fingerprint": self.elr_fingerprint,
                "config": self.config.elr.model_dump(),
            }
        elif self.pending_elr_state is not None:
            checkpoint["elr_state"] = self.pending_elr_state

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Defer history restoration until the training split is available."""
        saved = checkpoint.get("elr_state")
        if (self.config.elr is None) != (saved is None):
            raise ValueError("Checkpoint and configuration must agree on ELR")
        self.elr = None
        self.pending_elr_state = saved

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits."""
        return self.network(images)

    def shared_step(self, batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        """Compute classification loss and sample-weighted epoch metrics."""
        images, labels = batch[:2]
        logits = self(images)
        loss_config = self.config.loss
        if isinstance(loss_config, CrossEntropyConfig):
            loss = F.cross_entropy(logits, labels, label_smoothing=loss_config.label_smoothing)
        else:
            if logits.shape[1] != 2:
                raise ValueError("Binary losses require dataset.num_classes=2")
            binary_logits = logits[:, 1] - logits[:, 0]
            targets = labels.to(binary_logits)
            if isinstance(loss_config, BCEConfig):
                weight = None if loss_config.positive_class_weight is None else logits.new_tensor(loss_config.positive_class_weight)
                loss = F.binary_cross_entropy_with_logits(binary_logits, targets, pos_weight=weight)
            elif isinstance(loss_config, FocalConfig):
                loss = FocalLoss(loss_config.alpha, loss_config.gamma)(binary_logits, targets)
            else:
                loss = GeneralizedCrossEntropyLoss(loss_config.q)(binary_logits, targets)
        if stage == "train" and self.elr is not None and self.config.elr is not None:
            if len(batch) != 3:
                raise ValueError("ELR requires stable sample indices")
            regularization = self.elr(logits[:, 1] - logits[:, 0], batch[2])
            self.log("bce/train", loss, on_step=False, on_epoch=True, batch_size=len(labels))
            self.log("elr/train", regularization, on_step=False, on_epoch=True, batch_size=len(labels))
            loss = loss + self.config.elr.strength * regularization
        metrics = cast(MetricCollection, self.metrics[f"split_{stage}"])
        metrics.update(logits.softmax(dim=1), labels)
        self.log_dict(metrics, on_step=False, on_epoch=True)
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
        schedule = self.config.scheduler
        if schedule is None:
            return optimizer
        epochs = self.trainer.max_epochs
        if epochs is None or epochs <= schedule.warmup_epochs:
            raise ValueError("max_epochs must exceed scheduler.warmup_epochs")
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=schedule.start_factor, total_iters=schedule.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - schedule.warmup_epochs, eta_min=self.config.learning_rate * schedule.min_factor
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[schedule.warmup_epochs])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
