# Project instructions

Python ML research project. Use Python 3.11, the `src/` layout, pytest, Ruff, ty,
Pydantic, jsonargparse, and uv. Keep research code simple and readable.

Use short English sentences, active voice, and NumPy-style docstrings.
Document constructor arguments in `__init__`. Comments explain why.
Use modern type hints and specific names. Prefer direct code to abstractions.
Validate inputs at pipeline boundaries. Use standard Python logging.
Seed training and data loading.

Configuration models derive from `ConfigModel`. Keep them immutable, forbid
unknown fields, and validate defaults. Use typed nested models and readable YAML
with CLI overrides. Supply environment-specific paths through CLI arguments or
workflow scripts.

Do not modify input or processed datasets or run DVC unless asked. Keep generated
artifacts, credentials, and local paths out of Git. If a DVC pipeline is added,
edit `dvc.yaml` and let DVC generate its lock file.

Write a few tests for meaningful behavior. Keep the README brief and update it
when behavior or data contracts change.

Before completion, run:

- `uv run ruff check --fix .`
- `uv run ruff format --check .`
- `uv run ty check`
- `uv run pytest`

## Bootstrap a new project

1. Read `pyproject.toml`, the YAML examples, and this file. Rename the package and
   entry points if needed. Keep only the task-specific code the project needs.
2. Run `uv sync --locked` to create `.venv` from `uv.lock`. Use `uv run` for tools
   and commands. Use `uv add` to change dependencies and commit both project and
   lock files. Do not copy an environment from another project.
3. Check the PyTorch wheel indexes before changing CPU/CUDA support. The default
   Linux index uses CUDA 13.0. Keep torch and torchvision versions compatible.
4. Run configs/synthetic.yaml before connecting a dataset.
   Use CLI arguments for source paths. Never put site paths in model YAML.
5. Initialize DVC with `uv run dvc init` only when setting up the data workflow.
   Copy `paths.example.yaml` to ignored `paths.local.yaml` and ask for actual
   storage locations if they are unknown. Do not invent working paths.
6. Configure cache paths with `dvc cache dir --local` and remote credentials with
   local configuration. Shared caches need suitable group ownership and write
   permissions. Do not change shared-storage permissions without authorization.
7. Read `src/computer_vision/data_preparation/README.md`. Keep source datasets outside
   Git. Preparation writes `.artifacts/`; only `scripts/publish` writes published
   artifacts. Publishing requires an explicit request.
8. Use `uv run dvc repro --dry <target>` before large runs. Never fabricate
   artifacts or `dvc.lock`. Commit generated lock files only after successful
   reproduction. No lock file is supplied until the example has real inputs.

## Extension contracts

Keep the configurable ResNet blocks and typed loss variants. Binary BCE, focal,
and GCE use two-class logits through their difference. Cross entropy supports
multiple classes. Preserve numerical stability and checkpoint reconstruction
when changing losses or preprocessing. ELR tracks stable training-split indices
and saves prediction history. It runs on one device. Resume checks sample order, labels, split configuration, and ELR
settings. Do not use batch positions as sample IDs.

ONNX and torch.export artifacts contain the inference network. Stored image
channels use float intensities in [0, 1]. Inputs have a dynamic batch axis and
fixed spatial dimensions. Exported outputs are logits. Keep preprocessing and class ordering documented.
Test exported predictions against PyTorch when changing the network.
Torchvision adapters accept RGB tensors in [0, 1] and normalize inside the network.
Keep checkpoint restoration and training resumes free of ImageNet downloads.
Frozen backbones must keep running statistics fixed. ViT requires matching image size.
