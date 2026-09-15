"""Check the optional learning-rate schedule used by training and HPO."""

from typing import cast
from unittest.mock import Mock

import pytest
import torch
from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig

from computer_vision.lightning_module import ClassificationModule, CosineScheduleConfig, TrainingConfig
from computer_vision.model.resnet import ResNet, ResStageConfig


def test_warmup_cosine_and_constant_lr() -> None:
    """Reach the configured LR endpoints while preserving constant-LR behavior."""
    network = ResNet(in_channels=1, stages=(ResStageConfig(out_channels=2),))
    module = ClassificationModule(TrainingConfig(learning_rate=0.001), network=network)
    assert isinstance(module.configure_optimizers(), torch.optim.AdamW)
    module.config = TrainingConfig(learning_rate=0.001, scheduler=CosineScheduleConfig())
    module.trainer = Mock(max_epochs=20)
    configured = cast(OptimizerLRSchedulerConfig, module.configure_optimizers())
    optimizer = configured["optimizer"]
    schedule_config = cast(LRSchedulerConfigType, configured["lr_scheduler"])
    scheduler = schedule_config["scheduler"]
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(20):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] == pytest.approx(0.0001)
    assert rates[5] == pytest.approx(0.001)
    assert rates[-1] == pytest.approx(0.00001)
    assert rates[:6] == sorted(rates[:6])
    assert rates[5:] == sorted(rates[5:], reverse=True)
