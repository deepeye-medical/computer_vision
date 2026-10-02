"""Check training, checkpoint reconstruction, and reproducible folds."""

from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from pydantic import ValidationError

from computer_vision.data import DatasetConfig, ImageDataModule
from computer_vision.evaluate import load_checkpoint
from computer_vision.hpo import FloatSearchParameter, HPOConfig, StudyConfig, run_hpo
from computer_vision.lightning_module import ClassificationModule, ConfigCheckpoint
from computer_vision.train import TrainConfig, create_training_parser, dump_training_config, instantiate_training_config


@pytest.mark.parametrize("loss", ["cross_entropy", "bce", "focal", "gce"])
def test_checkpoint_roundtrip(tmp_path: Path, loss: str) -> None:
    """Fit synthetic images and reconstruct identical predictions from disk."""
    torch.set_num_threads(1)
    pl.seed_everything(42, workers=True)
    parser = create_training_parser(TrainConfig)
    classes = 3 if loss == "cross_entropy" else 2
    parsed = parser.parse_args(["--config", "configs/synthetic.yaml", f"--dataset.num_classes={classes}", "--training.loss", f"{{name: {loss}}}"])
    config, network = instantiate_training_config(parser, TrainConfig, parsed)
    module = ClassificationModule(config.training, network)
    checkpoint = ModelCheckpoint(dirpath=tmp_path, monitor="accuracy/validation", mode="max")
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        callbacks=[checkpoint, ConfigCheckpoint(dump_training_config(parser, parsed, config))],
    )
    trainer.fit(module, datamodule=ImageDataModule(config.dataset))
    restored, restored_config = load_checkpoint(Path(checkpoint.best_model_path))
    module.eval()
    restored.eval()
    images = torch.rand(2, 3, 32, 32)
    torch.testing.assert_close(module(images), restored(images))
    assert restored(images).shape == (2, classes)
    assert restored_config == config
    assert "accuracy/test" in trainer.test(restored, datamodule=ImageDataModule(config.dataset), verbose=False)[0]


def test_folds_and_validation() -> None:
    """Partition the source exactly once and reject misspelled configuration."""
    validation_indices = []
    for fold in range(3):
        module = ImageDataModule(DatasetConfig(validation_fold=fold))
        module.setup("fit")
        training = set(module.train_dataset.indices)
        validation = set(module.validation_dataset.indices)
        assert not training & validation
        assert training | validation == set(range(60))
        validation_indices.extend(validation)
    assert sorted(validation_indices) == list(range(60))
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"training": {"learning_rtae": 0.1}})


@pytest.mark.parametrize("tracking", [False, True])
def test_local_hpo(tmp_path: Path, tracking: bool) -> None:
    """Complete a local trial without remote experiment tracking."""
    torch.set_num_threads(1)
    training_path = tmp_path / "training.yaml"
    training_path.write_text(Path("configs/synthetic.yaml").read_text().replace("runs/training", str(tmp_path / "runs")))
    if tracking:
        with training_path.open("a") as stream:
            stream.write("\nwandb: {project: computer-vision-test, mode: offline}\n")
    config = HPOConfig(
        training_config=training_path,
        validation_folds=(0, 1) if tracking else (0,),
        study=StudyConfig(name="synthetic", storage=tmp_path / "study.log", n_trials=1),
        search_space=(FloatSearchParameter(path="training.learning_rate", low=0.0001, high=0.001),),
    )
    study = run_hpo(config)
    assert len(study.trials) == 1
    assert 0 <= study.best_value <= 1
    if tracking:
        assert len(list((tmp_path / "runs").rglob("run-*.wandb"))) == 2


def test_elr_resume_checks_membership(tmp_path: Path) -> None:
    """Restore ELR history and reject changed sample identity."""
    torch.set_num_threads(1)
    parser = create_training_parser(TrainConfig)
    parsed = parser.parse_args(["--config", "configs/synthetic.yaml", "--config", "configs/train_elr.yaml"])
    config, network = instantiate_training_config(parser, TrainConfig, parsed)
    module = ClassificationModule(config.training, network)
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False)
    trainer.fit(module, datamodule=ImageDataModule(config.dataset))
    checkpoint_path = tmp_path / "elr.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    _, restored_network = instantiate_training_config(parser, TrainConfig, parsed)
    resumed = ClassificationModule(config.training, restored_network)
    resumed_trainer = pl.Trainer(max_epochs=2, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False)
    resumed_trainer.fit(resumed, datamodule=ImageDataModule(config.dataset), ckpt_path=checkpoint_path)
    assert resumed.elr is not None
    assert resumed.elr.prediction_history.sum() > checkpoint["elr_state"]["prediction_history"].sum()
    _, changed_network = instantiate_training_config(parser, TrainConfig, parsed)
    changed = ClassificationModule(config.training, changed_network)
    changed_trainer = pl.Trainer(max_epochs=2, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False)
    with pytest.raises(ValueError, match="matching training membership"):
        changed_trainer.fit(changed, datamodule=ImageDataModule(config.dataset.model_copy(update={"split_seed": 7})), ckpt_path=checkpoint_path)
