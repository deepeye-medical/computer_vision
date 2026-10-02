"""Check offline tracking and cleanup without remote credentials."""

from pathlib import Path

import pytest
import wandb

from computer_vision.tracking import WandbConfig, wandb_run


@pytest.mark.parametrize("fail", [False, True])
def test_offline_run_cleanup(tmp_path: Path, fail: bool) -> None:
    """Record configuration and metrics, then close successful or failed runs."""

    def record_run() -> None:
        with wandb_run(WandbConfig(project="computer-vision-test", mode="offline"), tmp_path / "fold_0", {"seed": 42}, group="trial_0") as logger:
            assert logger is not None
            assert logger.experiment.group == "trial_0"
            assert logger.experiment.config["seed"] == 42
            logger.log_metrics({"accuracy/validation": 0.75}, step=1)
            if fail:
                raise RuntimeError("synthetic failure")

    if fail:
        with pytest.raises(RuntimeError, match="synthetic failure"):
            record_run()
    else:
        record_run()
    assert wandb.run is None
    assert len(list(tmp_path.rglob("run-*.wandb"))) == 1
