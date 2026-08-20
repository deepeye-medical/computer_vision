# Image preparation

Prepare two synthetic image sources, combine their manifests, and use the result
for training or viewing.

| Step | Input | Output |
| --- | --- | --- |
| `source_a/prepare.py` | Images and `labels.csv` with `path,label,split` and optional `group_id` | `.artifacts/source_a.csv` |
| `source_b/prepare.py` | `source/{train,test}/class/image` and optional `groups.csv` | `.artifacts/source_b.csv` |
| `integration.py` | Both source manifests | `.artifacts/images.csv` |
| `data_analysis/report.py` | Integrated manifest and source images | `.artifacts/report.json` |

Run the complete example without DVC:

```sh
uv run computer-vision-dummy
uv run computer-vision-prepare-a --source .artifacts/dummy/source_a
uv run computer-vision-prepare-b --source .artifacts/dummy/source_b
uv run computer-vision-integrate --manifests '[.artifacts/source_a.csv, .artifacts/source_b.csv]'
uv run computer-vision-report --manifest .artifacts/images.csv
uv run computer-vision-viewer --config configs/viewer.yaml
```

Open `http://127.0.0.1:8080`. The dummy generator rejects an existing output
directory to avoid mixing runs. Use `--output` to select a new directory.

Train on the same manifest:

```sh
uv run computer-vision-train --config configs/train_manifest.yaml
uv run computer-vision-hpo --config configs/hpo_manifest.yaml --local
```

## Manifest contract

Each row has `sample_id,source,path,label,split,width,height,group_id`.
Sample IDs include the source name and source-relative image path. Paths are relative to the CSV's
parent directory. Preparation checks images without changing them. Integration
rebases paths and rejects repeated IDs or resolved image paths. The report also
counts duplicate decoded RGB content, including train/test duplication, missing
files, missing labels, and groups by split.
Source labels must use common class names; adapt source preparation when needed.

Splits are `train` or `test`. Training divides train rows into deterministic
group folds. Groups sort by decreasing image count, with ID as the tie-breaker,
and receive folds in round-robin order. All images from a group stay together.
Sorted class names define the shared class indices. The loader
rejects empty labels, repeated IDs, and a class count that differs from configuration.
Source A reads groups from `labels.csv`. Source B reads an optional `groups.csv`
with `path,group_id`; when supplied, it must cover every image. Without group
metadata, each sample forms its own group. Reuse group IDs across sources only
for the same entity. Integration and loading reject train/test group overlap.
Group folds ignore `split_seed` and do not stratify labels. Set
`dataset.fold_strategy: random` to use seeded image folds instead.
Integration does not check duplicate image content. Review the report for
train/test duplicates before training.

Manifests reference source images; they do not contain them. Moving a manifest
requires rebasing paths. Moving the whole directory tree preserves relative paths.
The `computer-vision-prepare` command uses the same class-folder adapter and manifest
contract as `computer-vision-prepare-b`, with `.artifacts/images.csv` as its default output.

## DVC and publishing

Copy `paths.example.yaml` to ignored `paths.local.yaml` and set `source_a_dir`
and `source_b_dir`. Keep real source datasets outside Git. The generator is a
local setup tool, separate from DVC. Source preparation, integration, and analysis
write to `.artifacts/`. The viewer runs separately and does not change images.

Initialize DVC once with `uv run dvc init`. Configure a shared cache with
`uv run dvc cache dir --local /path/to/shared/cache`. Store credentials in local
configuration. Never add source datasets with `dvc add`. For large sources,
use a maintained source-version manifest instead of hashing the full directory.

Run `uv run dvc repro --dry analyze_images` before a large run. `scripts/repro`
checks out cached artifacts and reproduces stages. DVC creates `dvc.lock`;
commit it with the code and YAML after successful reproduction. Do not edit it.

Publishing requires an explicit request. Set `PUBLISH_DIR` and run
`scripts/publish`. It runs reproduction, writes the three manifests with paths
rebased to the destination, and copies the report. Source images must remain
available. Publishing stays outside the DVC pipeline.
