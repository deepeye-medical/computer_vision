"""Check CNN classification across input layouts."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from computer_vision.model import CNN, CNNStage, CNNStageConfig


@pytest.mark.parametrize("mode", ["image", "slice", "volume"])
def test_cnn_modes_train(mode: str) -> None:
    """Train with one sample and check the feature stage order."""
    torch.manual_seed(42)
    torch.set_num_threads(1)
    dimensions = 3 if mode == "volume" else 2
    network = CNN(
        in_channels=1,
        stages=(
            CNNStageConfig(
                out_channels=4, kernel_size=(3,) * dimensions, stride=(1,) * dimensions, padding=(0,) * dimensions, pool_size=(2,) * dimensions
            ),
        ),
        slice_wise=mode == "slice",
        slice_pooling="mean_max",
        use_coordinate_channels=True,
    )
    images = torch.rand((1, 1, 16, 16) if mode == "image" else (1, 1, 4, 16, 16))
    logits = network(images)
    loss = F.cross_entropy(logits, torch.tensor([1]))
    loss.backward()
    assert logits.shape == (1, 2)
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in network.parameters())
    stage = network.stages[0]
    assert isinstance(stage, CNNStage)
    assert isinstance(stage[1], nn.ReLU)
    assert isinstance(stage[2], (nn.BatchNorm2d, nn.BatchNorm3d))
    assert isinstance(stage[3], (nn.MaxPool2d, nn.MaxPool3d))


def test_slice_cnn_joins_features_before_head() -> None:
    """Combine independently encoded slice features before classification."""
    network = CNN(stages=(CNNStageConfig(out_channels=4),), slice_wise=True, dropout=0).eval()
    images = torch.rand(2, 3, 4, 16, 16)
    with torch.no_grad():
        features = network.stages(images.permute(0, 2, 1, 3, 4).reshape(8, 3, 16, 16)).mean(dim=(2, 3)).reshape(2, 4, -1)
        torch.testing.assert_close(network(images), network.classifier(features.mean(dim=1)))
