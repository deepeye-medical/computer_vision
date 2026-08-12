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

