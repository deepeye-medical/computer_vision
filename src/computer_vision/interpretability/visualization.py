"""Save attribution arrays and input, heatmap, and overlay panels."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from computer_vision.interpretability.attribution import AttributionConfig, AttributionResult


def save_visualizations(
    images: torch.Tensor,
    result: AttributionResult,
    config: AttributionConfig,
    output_dir: Path,
    class_names: tuple[str, ...] | None = None,
) -> None:
    """Save raw maps and PNG panels with one common scale per sample.

    Parameters
    ----------
    images
        Original batched stored-channel inputs in [0, 1].
    result
        Attribution tensors on the input grid.
    config
        Method settings saved alongside the arrays.
    output_dir
        Destination for maps.npz and per-sample, per-depth PNG files.
    class_names
        Optional checkpoint class ordering.
    """
    if images.ndim not in (4, 5) or result.maps.shape != (images.shape[0], 1, *images.shape[2:]):
        raise ValueError("Attribution maps must match the batched input grid")
    output_dir.mkdir(parents=True, exist_ok=True)
    completeness = result.completeness_error
    np.savez_compressed(
        output_dir / "maps.npz",
        attribution=result.attribution.cpu().numpy(),
        maps=result.maps.cpu().numpy(),
        logits=result.logits.cpu().numpy(),
        targets=result.targets.cpu().numpy(),
        completeness_error=np.array([]) if completeness is None else completeness.cpu().numpy(),
        config=np.array(config.model_dump_json()),
        class_names=np.array(class_names or (), dtype=str),
    )
    signed = config.method in ("integrated_gradients", "occlusion")
    for sample_index, (sample, sample_maps) in enumerate(zip(images.detach().cpu(), result.maps.cpu(), strict=True)):
        if sample.shape[0] == 3:
            display = sample
            channel_label = "RGB"
        else:
            display = sample.mean(dim=0, keepdim=True).expand(3, *sample.shape[1:])
            channel_label = "channel mean"
        scale = sample_maps.abs().amax()
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        normalized = sample_maps[0] / scale
        if sample.ndim == 3:
            display = display.unsqueeze(1)
            normalized = normalized.unsqueeze(0)
        class_index = int(result.targets[sample_index])
        target = str(class_index) if class_names is None else class_names[class_index]
        for depth_index in range(display.shape[1]):
            intensity = normalized[depth_index]
            if signed:
                # Red and blue use the same magnitude scale for opposite signs.
                heat = torch.stack((intensity.clamp_min(0), torch.zeros_like(intensity), (-intensity).clamp_min(0)))
            else:
                heat = torch.stack((intensity, intensity.square(), torch.zeros_like(intensity)))
            original = display[:, depth_index].clamp(0, 1)
            alpha = 0.6 * intensity.abs().unsqueeze(0)
            overlay = original * (1 - alpha) + heat * alpha
            panels = torch.cat((original, heat, overlay), dim=2).permute(1, 2, 0)
            pixels = (panels.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
            height, width = pixels.shape[:2]
            canvas = Image.new("RGB", (max(width, 480), height + 48), "white")
            canvas.paste(Image.fromarray(pixels), (0, 48))
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 3), f"{config.method}: class {target} | sample {sample_index}, depth {depth_index}", fill="black")
            if signed:
                legend = "red + / blue -"
            elif config.method == "saliency":
                legend = "red to yellow: gradient magnitude"
            else:
                legend = "red to yellow: positive contribution"
            draw.text((4, 19), f"input ({channel_label}) | map | overlay; {legend}", fill="black")
            draw.text((4, 33), f"map scale: {float(scale):.4g}; same scale across depth", fill="black")
            canvas.save(output_dir / f"sample-{sample_index:03d}-depth-{depth_index:03d}.png")
