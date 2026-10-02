"""Optional experiment tracking shared by training and HPO."""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from lightning.pytorch.loggers import WandbLogger
from pydantic import Field

from computer_vision.config import ConfigModel


class WandbConfig(ConfigModel):
    """Settings for optional Weights & Biases tracking."""

    project: str = Field(min_length=1)
    """Project receiving metrics and the resolved training configuration."""

    entity: str | None = None
    """Team or user owning the project; null uses the authenticated account."""

    mode: Literal["online", "offline"] = "online"
    """Upload during training or keep a local run for later synchronization."""


@contextmanager
def wandb_run(
    config: WandbConfig | None,
    run_dir: Path,
    resolved_config: dict[str, object],
    group: str | None = None,
) -> Iterator[WandbLogger | None]:
    """Create a logger and finish its run on success or failure.

    Parameters
    ----------
    config
        Tracking settings, or null to disable W&B.
    run_dir
        Local output directory; its name identifies the run.
    resolved_config
        Complete training settings, including the selected network.
    group
        Trial identifier shared by HPO fold runs.
    """
    if config is None:
        yield None
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = WandbLogger(
        project=config.project,
        entity=config.entity,
        mode=config.mode,
        name=run_dir.name,
        group=group,
        save_dir=run_dir,
        log_model=False,
    )
    experiment = logger.experiment
    try:
        logger.log_hyperparams(resolved_config)
        yield logger
    finally:
        experiment.finish(exit_code=int(sys.exc_info()[0] is not None))
