# computer_vision

Computer vision research with PyTorch.

## Setup

```sh
uv sync
```

## Data preparation

Supply source paths through local DVC variables. Preparation writes cached
artifacts under `.artifacts/`. Run `scripts/repro` to restore and reproduce them.

## Interpretability

Explain a checkpoint with Grad-CAM and save raw maps plus PNG overlays:

```sh
uv run computer-vision-explain --checkpoint runs/training/<run-id>/checkpoints/best.ckpt \
  --input example.png --output_dir runs/explanations/example
```

Use `--attribution.method saliency`, `integrated_gradients`, or `occlusion` for
other methods. `--attribution.target_class 1` selects a class; the default explains
the predicted class. All methods support images, independent slices, and volumes.
The command accepts RGB image files or channel-first float `.npy` arrays in [0, 1].
See [attribution settings and map meanings](docs/interpretability.md).
