# Project architecture

| Component | Behavior |
| --- | --- |
| Configuration | Immutable Pydantic models validate defaults and reject unknown fields. jsonargparse loads YAML and CLI overrides. |
| Training | Lightning seeds model initialization and workers. AdamW supports warmup and cosine decay. Checkpoints use validation accuracy. |
| Tracking | TensorBoard saves local logs. ClearML is optional. Python logging reports runtime events. |
| HPO | Optuna searches scalar settings and YAML choices, compares fold means, and stores trials in a resumable journal. Workers run locally or on Slurm. |
| Models | CNN, ResNet, and ViT support images, independent slices, volumes, and coordinates. See [model settings](models.md). |
| Losses | Cross entropy supports multiple classes. BCE, focal loss, GCE, and ELR use two-class logits. |
| Checkpoints | Saved settings reconstruct the network without pretrained downloads. Class names preserve output order. ELR checks training membership on resume. |
| Exports | Weights, torch.export, or ONNX contain the inference network. Inputs have a dynamic batch axis; outputs are logits. |
| Attribution | Grad-CAM, saliency, integrated gradients, and occlusion save raw maps and PNG overlays. |
| Preparation | Source adapters write manifests under `.artifacts/`; integration combines them. Source images stay outside Git. |
| Analysis | The report counts missing records and duplicate image content. The viewer uses the training loader, folds, and augmentation. |
| Folds | Manifest groups sort by size and ID, then receive round-robin folds. Related images stay together. Other loaders use seeded image folds. |
| Augmentation | Kornia applies spatial and intensity transforms to training batches. Evaluation uses deterministic preprocessing. |
| DVC | `dvc.yaml` defines stages and artifacts. DVC generates `dvc.lock` after reproduction. Paths and credentials use local settings. |
| Publishing | `scripts/repro` restores and reproduces cached artifacts. `scripts/publish` copies them to a chosen destination and rebases manifest paths. |
| Tests | Generated inputs require no remote services. CI runs Ruff, ty, and pytest. |

Use a shared `group_id` for related images. Group IDs must identify the same
entity across sources. Train and test groups must be disjoint. Group folds do
not stratify labels or guarantee equal image counts. Empty IDs treat samples
as independent.

Pretrained networks normalize stored channels internally. Coordinates keep
their signed range; resizing stays in the loader. ViT requires its configured
grid. Slice and volume modes need matching loaders, augmentation, and export
shapes. Checkpoints store full Python class paths; update saved paths after a
package rename before loading older checkpoints.

Keep source datasets, generated artifacts, credentials, and run outputs outside
Git. See [preparation setup](../src/computer_vision/data_preparation/README.md)
and [agent instructions](../AGENTS.md) for cache and publishing rules.
