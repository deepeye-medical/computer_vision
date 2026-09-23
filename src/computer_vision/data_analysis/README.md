# Training data analysis

```sh
uv run computer-vision-report --manifest .artifacts/images.csv
uv run computer-vision-viewer --config configs/viewer.yaml
```

The JSON report counts classes, sources, splits, recorded image sizes, missing
labels and files, repeated sample IDs, groups by split, and duplicate decoded RGB
images. It reports train/test duplicate content separately.

The viewer runs at `http://127.0.0.1:8080`. Filter by source, class, or train/validation/test
split. Use Previous, Next, or the zero-based position field to select an image.
The page shows metadata, the original RGB image, the resized model input, and a
seeded augmentation preview. It uses the training dataset, fold assignment, and
augmentation code. Training and validation filters select the exact subsets for
`dataset.validation_fold`. The page displays the resolved dataset settings.

Use the same dataset configuration as training, including `dataset.augmentation`,
`dataset.fold_strategy`, and fold settings. Apply the same YAML overlays to both
commands, for example `--config configs/augmentation_mild.yaml`.
Override the manifest with `--dataset.root /path/to/images.csv`.

Load a training checkpoint for CPU inference on the current image:

```sh
uv run computer-vision-viewer --config configs/viewer.yaml --checkpoint runs/training/<run-id>/checkpoints/best.ckpt
```

The model loads once at startup. It uses the checkpoint's saved preprocessing,
augmentation settings, and validation fold, with the viewer's manifest path.
Scores use the deterministic model input, not the augmentation preview.
The page shows the predicted class and softmax score for each class.
Checkpoints save class ordering; the viewer rejects a different manifest class set.
Older checkpoints without class names display output indices instead.
The viewer is read-only.
