"""Class-logit attribution for image, slice, and volume classifiers."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import nn
from torchvision import models

from computer_vision.config import ConfigModel
from computer_vision.model.cnn import CNN
from computer_vision.model.pretrained import TorchvisionClassifier
from computer_vision.model.resnet import ResNet
from computer_vision.model.vit import ViT

Method = Literal["gradcam", "saliency", "integrated_gradients", "occlusion"]


class AttributionConfig(ConfigModel):
    """Settings for attribution of a fixed class logit."""

    method: Method = "gradcam"
    target_class: int | None = Field(default=None, ge=0)
    """Class index; null selects each sample's original predicted class."""
    target_layer: str | None = None
    """Grad-CAM module path; null uses the model's default feature layer."""
    steps: int = Field(default=32, ge=1)
    """Trapezoidal integration intervals for integrated gradients."""
    baseline: float = Field(default=0.0, ge=0.0, le=1.0)
    """Stored-channel intensity for the IG reference or occluded region."""
    occlusion_size: int = Field(default=8, ge=1)
    """Occlusion window width on every spatial axis, including depth."""
    occlusion_stride: int = Field(default=8, ge=1)
    """Occlusion stride, capped at the window size to cover all positions."""


@dataclass(frozen=True)
class AttributionResult:
    """Raw attribution, spatial maps, and the fixed prediction target."""

    logits: torch.Tensor
    targets: torch.Tensor
    attribution: torch.Tensor
    maps: torch.Tensor
    completeness_error: torch.Tensor | None = None


@contextmanager
def evaluation_mode(network: nn.Module) -> Iterator[None]:
    """Use evaluation behavior and restore every module's previous mode.

    Parameters
    ----------
    network
        Network whose dropout and running statistics must stay fixed.
    """
    modes = [(module, module.training) for module in network.modules()]
    network.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def gradcam_layer(network: nn.Module) -> tuple[nn.Module, tuple[int, ...] | None]:
    """Select a spatial feature layer or pre-attention ViT token layer.

    Parameters
    ----------
    network
        Supported classifier. Custom networks must supply a layer path.

    Returns
    -------
    tuple
        Module and optional token grid in spatial tensor order.
    """
    if isinstance(network, (CNN, ResNet)):
        return network.stages[-1], None
    if isinstance(network, ViT):
        return network.encoder.layers[-1].get_submodule("ln_1"), network.patch_grid
    if isinstance(network, TorchvisionClassifier):
        if isinstance(network.backbone, models.ResNet):
            return network.backbone.layer4, None
        if isinstance(network.backbone, models.VisionTransformer):
            backbone = network.backbone
            size = backbone.image_size // backbone.patch_size
            return backbone.encoder.layers[-1].get_submodule("ln_1"), (size, size)
    raise ValueError("Supply target_layer for Grad-CAM on a custom network")


@torch.enable_grad()
def attribute(network: nn.Module, images: torch.Tensor, config: AttributionConfig | None = None) -> AttributionResult:
    """Explain fixed class logits without changing parameter gradients or modes.

    Parameters
    ----------
    network
        Classifier returning (batch, classes) logits. Samples must be independent
        in evaluation mode. Stored-channel normalization stays in the network.
    images
        Float inputs in [0, 1], shaped (B, C, H, W) or (B, C, D, H, W).
    config
        Method settings and optional class and feature-layer selection.

    Returns
    -------
    AttributionResult
        Detached tensors. Maps have shape (B, 1, *spatial_shape). Saliency maps
        sum absolute channel gradients; IG maps sum signed channel contributions.
        Occlusion maps average signed logit drops over overlapping windows.
        Grad-CAM maps retain positive contributions without display normalization.
        IG also reports attribution sum minus the input-to-baseline logit change.
    """
    config = config or AttributionConfig()
    if torch.is_inference_mode_enabled():
        raise ValueError("Attribution needs autograd; leave torch.inference_mode() first")
    if images.ndim not in (4, 5) or any(size == 0 for size in images.shape) or not images.is_floating_point():
        raise ValueError("Expected nonempty float inputs shaped (B, C, H, W) or (B, C, D, H, W)")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Inputs must contain finite stored-channel intensities in [0, 1]")
    inputs = images.detach().clone().requires_grad_(True)
    with evaluation_mode(network):
        layer = None
        token_grid = None
        handle = None
        captured: list[torch.Tensor] = []
        if config.method == "gradcam":
            if config.target_layer is None:
                layer, token_grid = gradcam_layer(network)
            else:
                layer = network.get_submodule(config.target_layer)
                if isinstance(network, ViT):
                    token_grid = network.patch_grid
                elif isinstance(network, TorchvisionClassifier) and isinstance(network.backbone, models.VisionTransformer):
                    size = network.backbone.image_size // network.backbone.patch_size
                    token_grid = (size, size)

            def capture(_module: nn.Module, _arguments: tuple[object, ...], output: torch.Tensor) -> torch.Tensor:
                if not isinstance(output, torch.Tensor):
                    raise ValueError("Grad-CAM target layer must return a tensor")
                captured.append(output)
                # Protect the captured activation from downstream in-place ReLUs.
                return output.clone()

            handle = layer.register_forward_hook(capture)
        try:
            logits = network(inputs)
            if logits.ndim != 2 or logits.shape[0] != inputs.shape[0] or logits.shape[1] < 2:
                raise ValueError("Network must return (batch, classes) logits with at least two classes")
            targets = logits.argmax(dim=1) if config.target_class is None else torch.full_like(logits.argmax(dim=1), config.target_class)
            if (targets >= logits.shape[1]).any():
                raise ValueError("target_class exceeds the network class count")
            scores = logits.gather(1, targets[:, None]).sum()
            completeness_error = None
            if config.method == "gradcam":
                if len(captured) != 1:
                    raise ValueError("Grad-CAM target layer must execute exactly once per forward pass")
                features = captured[0]
                gradients = torch.autograd.grad(scores, features)[0]
                if features.ndim == 3:
                    if token_grid is None or features.shape[1] != 1 + prod(token_grid):
                        raise ValueError("Token Grad-CAM requires a supported ViT grid and one class token")
                    features = features[:, 1:].transpose(1, 2).reshape(features.shape[0], -1, *token_grid)
                    gradients = gradients[:, 1:].transpose(1, 2).reshape_as(features)
                if features.ndim not in (4, 5):
                    raise ValueError("Grad-CAM requires spatial features or ViT patch tokens")
                weights = gradients.mean(dim=tuple(range(2, gradients.ndim)), keepdim=True)
                maps = (weights * features).sum(dim=1, keepdim=True).relu()
                if getattr(network, "slice_wise", False):
                    batch_size, _, depth, height, width = inputs.shape
                    maps = F.interpolate(maps, size=(height, width), mode="bilinear", align_corners=False)
                    maps = maps.reshape(batch_size, depth, 1, height, width).permute(0, 2, 1, 3, 4)
                else:
                    mode = "bilinear" if images.ndim == 4 else "trilinear"
                    maps = F.interpolate(maps, size=images.shape[2:], mode=mode, align_corners=False)
                attribution = maps
            elif config.method == "saliency":
                attribution = torch.autograd.grad(scores, inputs)[0]
                maps = attribution.abs().sum(dim=1, keepdim=True)
            elif config.method == "integrated_gradients":
                reference = torch.full_like(inputs, config.baseline)
                difference = inputs.detach() - reference
                accumulated = torch.zeros_like(inputs)
                for step in range(config.steps + 1):
                    interpolated = (reference + difference * (step / config.steps)).requires_grad_(True)
                    step_scores = network(interpolated).gather(1, targets[:, None]).sum()
                    gradient = torch.autograd.grad(step_scores, interpolated)[0]
                    accumulated += gradient * (0.5 if step in (0, config.steps) else 1.0)
                attribution = difference * accumulated / config.steps
                maps = attribution.sum(dim=1, keepdim=True)
                with torch.no_grad():
                    logit_change = (logits - network(reference)).gather(1, targets[:, None]).squeeze(1)
                completeness_error = attribution.flatten(1).sum(dim=1) - logit_change
            else:
                maps = torch.zeros_like(inputs[:, :1])
                counts = torch.zeros_like(maps)
                windows = [min(config.occlusion_size, size) for size in inputs.shape[2:]]
                stride = min(config.occlusion_stride, config.occlusion_size)
                starts = [
                    sorted({*range(0, size - window + 1, stride), size - window}) for size, window in zip(inputs.shape[2:], windows, strict=True)
                ]
                original_scores = logits.detach().gather(1, targets[:, None])
                with torch.no_grad():
                    for position in product(*starts):
                        region = (slice(None), slice(None), *(slice(start, start + window) for start, window in zip(position, windows, strict=True)))
                        occluded = inputs.detach().clone()
                        occluded[region] = config.baseline
                        drop = original_scores - network(occluded).gather(1, targets[:, None])
                        maps[region] += drop.reshape(inputs.shape[0], 1, *([1] * (inputs.ndim - 2)))
                        counts[region] += 1
                maps /= counts
                attribution = maps
        finally:
            if handle is not None:
                handle.remove()
    return AttributionResult(
        logits.detach(),
        targets.detach(),
        attribution.detach(),
        maps.detach(),
        None if completeness_error is None else completeness_error.detach(),
    )
