# Interpretability

Explain a checkpoint prediction and save attribution maps with PNG overlays:

```sh
uv run computer-vision-explain --checkpoint model.ckpt --input image.png \
  --output_dir runs/explanations --attribution.method gradcam
```

The command restores the network and normalization and runs in evaluation mode.
It explains the predicted class by default. Use `--attribution.target_class 1`
to select a class. The class stays fixed throughout the calculation.

RGB image files use the checkpoint's resize settings. For non-RGB inputs, slices,
or volumes, use a float `.npy` array shaped `(C, H, W)` or `(C, D, H, W)`.
Values must be in [0, 1]. NumPy inputs receive no resize; supply the network's
required grid. CPU is the default; `--device cuda` or `mps` selects another device.

| Method | Map meaning | Options under `--attribution.*` |
| --- | --- | --- |
| `gradcam` | Positive feature contributions to the class score. | `target_layer` overrides the feature layer. |
| `saliency` | Absolute input-gradient magnitude, summed across channels. | `target_class` selects the class. |
| `integrated_gradients` | Signed contributions relative to a reference intensity. | `steps` (32) and `baseline` (0). |
| `occlusion` | Signed score drop when replacing a region. | `occlusion_size` (8), `occlusion_stride` (8), and `baseline` (0). |

Grad-CAM uses the last CNN/ResNet stage or ViT features before the final attention
block. Custom networks need a `target_layer`. Slice models explain the shared
class score on each slice; volume models return a 3D map. Increase integration
steps to check convergence. Occlusion needs one forward pass per window.

`maps.npz` contains raw attribution values, spatial maps, logits, class indices,
class names, settings, and the integrated-gradients completeness error. This error
compares the attribution sum with the input-to-reference score change. Load the
archive with `numpy.load(path, allow_pickle=False)`.

Each PNG shows the input, heatmap, and overlay. Signed maps use red for positive
values and blue for negative values. All depth slices share one display scale;
raw arrays retain their values. Non-RGB displays use the channel mean.
Use `--help` for all settings.
