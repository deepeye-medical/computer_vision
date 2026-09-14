"""Tests for classification loss wrappers."""

import pytest
import torch

from computer_vision.loss import FocalLoss


@pytest.fixture
def binary_logits_and_targets() -> tuple[torch.Tensor, torch.Tensor]:
    """Return representative logits and binary targets."""
    logits = torch.tensor([[-1.0], [1.0]], requires_grad=True)
    targets = torch.tensor([[0.0], [1.0]])
    return logits, targets


def test_focal_loss_supports_backpropagation(
    binary_logits_and_targets: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify that focal loss supports backpropagation."""
    logits, targets = binary_logits_and_targets

    loss = FocalLoss(alpha=0.25, gamma=2.0)(logits, targets)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()




