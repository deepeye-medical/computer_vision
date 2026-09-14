"""Tests for classification loss wrappers."""

import pytest
import torch

from computer_vision.loss import ELRRegularization, FocalLoss


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


def test_elr_updates_indexed_history_and_matches_binary_formula() -> None:
    """Track repeated examples across shuffled batches without differentiating history."""
    regularizer = ELRRegularization(num_samples=4, beta=0.7)
    expected_history = torch.zeros(4, 2)
    for indices, values in [([2, 0], [-1.0, 2.0]), ([1, 2], [0.5, -2.0])]:
        logits = torch.tensor(values, requires_grad=True)
        sample_indices = torch.tensor(indices)
        probabilities = torch.stack((1 - logits.sigmoid(), logits.sigmoid()), dim=1)
        expected_history[sample_indices] = 0.7 * expected_history[sample_indices] + 0.3 * probabilities.detach()
        expected = torch.log(1 - (expected_history[sample_indices] * probabilities).sum(dim=1)).mean()

        actual = regularizer(logits, sample_indices)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(regularizer.prediction_history, expected_history)
        expected_gradient = torch.autograd.grad(expected, logits)[0]
        actual.backward()
        torch.testing.assert_close(logits.grad, expected_gradient)
        assert not regularizer.prediction_history.requires_grad
    torch.testing.assert_close(regularizer.prediction_history.sum(dim=1), torch.tensor([0.3, 0.3, 0.51, 0.0]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_elr_is_finite_for_saturated_predictions(dtype: torch.dtype) -> None:
    """Keep logarithms and gradients finite even after a long confident history."""
    regularizer = ELRRegularization(num_samples=2, beta=0.7)
    regularizer.prediction_history.copy_(torch.eye(2))
    logits = torch.tensor([-1000.0, 1000.0], dtype=dtype, requires_grad=True)
    loss = regularizer(logits, torch.arange(2))
    loss.backward()
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
