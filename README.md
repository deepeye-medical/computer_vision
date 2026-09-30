# computer_vision

Python 3.11 code for computer vision research with PyTorch and Lightning.
Includes typed configuration, training, HPO, tracking, exports, and image preparation.

## Setup

```sh
uv sync --locked
uv run computer-vision-train --config configs/synthetic.yaml
uv run computer-vision-test --checkpoint runs/training/<run-id>/checkpoints/best.ckpt --accelerator cpu
```

The synthetic example uses generated RGB images and random labels to check the
training pipeline. It needs no downloads or tracking account. Linux uses CUDA
13.0 PyTorch wheels; other platforms use the CPU index.

## Data and models

Put images under `<root>/train/<class>/` and `<root>/test/<class>/`.
Both splits must use the same class names:

```sh
uv run computer-vision-train --config configs/train.yaml \
  --dataset.kind imagefolder --dataset.root /path/to/images \
  --dataset.num_classes 10 --dataset.image_size 128
```

For CSV manifests, select `--dataset.kind manifest` and pass the manifest path
with `--dataset.root`. See [image preparation](src/computer_vision/data_preparation/README.md)
and the [dataset viewer](src/computer_vision/data_analysis/README.md).

Manifest folds keep images with the same `group_id` together. Groups sort by
image count and ID, then receive folds in round-robin order. This does not
stratify labels. Train and test groups must be disjoint. Set
`dataset.fold_strategy: random` for seeded image folds, as used by ImageFolder
and generated images. `split_seed` controls folds; `seed` controls training.

The loader supplies RGB tensors in [0, 1]. Training applies augmentation on the
device; validation and test use deterministic preprocessing. Configure spatial
and intensity transforms under `dataset.augmentation`, or add
`--config configs/augmentation_mild.yaml`. Choose transforms that preserve labels.

CNN, ResNet, and ViT support images, independent slices, and volumes. Slice and
volume modes require a custom loader and augmentation code. Add a model overlay:

```sh
uv run computer-vision-train --config configs/train_manifest.yaml --config configs/model_cnn.yaml
uv run computer-vision-train --config configs/train_manifest.yaml --config configs/model_imagenet_resnet.yaml
uv run computer-vision-train --config configs/train_manifest.yaml --config configs/model_imagenet_vit.yaml
```

See [model settings](docs/models.md) for pooling, coordinates, pretrained weights,
and input sizes. [Project architecture](docs/architecture.md) describes the modules.

Losses include cross entropy, binary BCE, focal loss, and GCE. Add a loss overlay
such as `--config configs/loss_focal.yaml`. Binary losses require two class logits.
`configs/train_elr.yaml` enables ELR with BCE on one device. Resume ELR with
`--resume_from` and unchanged training membership, labels, and ELR settings.
AdamW supports warmup and cosine decay.

## Tracking and HPO

Runs save TensorBoard logs, resolved YAML, and best/last checkpoints under
`output_dir`. Checkpoints include model settings and class ordering.
Set `clearml_project_name` to enable ClearML with your configured credentials.

```sh
uv run tensorboard --logdir runs
uv run computer-vision-hpo --config configs/hpo.yaml --local
uv run computer-vision-hpo --config configs/hpo.yaml
```

The last command submits a Slurm array. Set resources in the YAML first.
Workers must share image paths and the Optuna journal. Trials evaluate the same
folds and optimize their mean best validation score, with pruning between folds.
Search parameters can be scalars or named YAML overlays. Existing studies resume;
`n_trials` sets the additional trial count per invocation.

The manifest example uses `configs/hpo_manifest.yaml`. Reported metrics include
accuracy, macro AUROC, balanced accuracy, precision, recall, and specificity.
Training selects checkpoints by validation accuracy.

## Export and attribution

```sh
uv run computer-vision-export --checkpoint model.ckpt --output runs/model.onnx
uv run computer-vision-explain --checkpoint model.ckpt --input example.png --output_dir runs/explanations
```

Exports accept `onnx`, `torch_export`, or `weights` through `--format`. Inference
inputs have a dynamic batch axis and fixed spatial dimensions; outputs are logits.
Pretrained normalization stays in the network. Resizing stays in the loader.
ViT uses its configured grid. Slice and CNN/ResNet volume exports require
`--input_shape '[channels, depth, height, width]'`. Weight exports need no shape.

Attribution methods include Grad-CAM, saliency, integrated gradients, and occlusion.
Outputs contain raw arrays and PNG overlays. See [interpretability](docs/interpretability.md).

## Development

Use `--help` for command options and `--print_config` for training settings.
Regenerate the reference YAML with `uv run python scripts/generate_config_examples.py`;
add `--check` to verify it. Add loaders in `src/computer_vision/data.py` and networks
under `src/computer_vision/model/`.

```sh
uv run ruff check --fix .
uv run ruff format --check .
uv run ty check
uv run pytest
```

Preparation writes cached artifacts to `.artifacts/`. `scripts/repro` restores
and reproduces them; `scripts/publish` copies them to an explicit destination.
See [preparation setup](src/computer_vision/data_preparation/README.md) and
[agent instructions](AGENTS.md) before connecting source datasets or shared storage.
