"""Standalone Optuna hyperparameter tuning for image classification training."""

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from itertools import product
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Annotated, Literal, TypeAlias, cast

import lightning.pytorch as pl
import numpy as np
import optuna
import yaml
from clearml import Task
from clearml.config import config as clearml_config
from coolname import generate_slug
from jsonargparse import Namespace
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from pydantic import Field, model_validator

from computer_vision.config import ConfigModel, parse_and_validate
from computer_vision.data import ImageDataModule
from computer_vision.lightning_module import ClassificationModule, ConfigCheckpoint
from computer_vision.train import (
    TrainConfig,
    create_training_parser,
    dump_training_config,
    instantiate_training_config,
)

logger = logging.getLogger(__name__)


class FloatSearchParameter(ConfigModel):
    """Continuous floating-point search parameter."""

    path: str = Field(min_length=1)
    """Dot-separated path in the image classification training configuration."""

    type: Literal["float"] = "float"
    """Optuna distribution type."""

    low: float
    """Inclusive lower bound."""

    high: float
    """Inclusive upper bound."""

    log: bool = False
    """Whether to sample on a logarithmic scale."""

    step: float | None = Field(default=None, gt=0.0)
    """Optional sampling interval."""

    @model_validator(mode="after")
    def validate_range(self) -> "FloatSearchParameter":
        """Reject empty ranges and incompatible Optuna options."""
        if self.low >= self.high:
            raise ValueError("low must be smaller than high")
        if self.log and self.low <= 0:
            raise ValueError("a logarithmic range must have low > 0")
        if self.log and self.step is not None:
            raise ValueError("step cannot be used with a logarithmic range")
        return self


class IntSearchParameter(ConfigModel):
    """Discrete integer search parameter."""

    path: str = Field(min_length=1)
    """Dot-separated path in the image classification training configuration."""

    type: Literal["int"] = "int"
    """Optuna distribution type."""

    low: int
    """Inclusive lower bound."""

    high: int
    """Inclusive upper bound."""

    step: int = Field(default=1, gt=0)
    """Sampling interval."""

    log: bool = False
    """Whether to sample on a logarithmic scale."""

    @model_validator(mode="after")
    def validate_range(self) -> "IntSearchParameter":
        """Reject empty ranges and incompatible Optuna options."""
        if self.low >= self.high:
            raise ValueError("low must be smaller than high")
        if self.log and self.low <= 0:
            raise ValueError("a logarithmic range must have low > 0")
        if self.log and self.step != 1:
            raise ValueError("step must be 1 with a logarithmic range")
        return self


SearchChoice: TypeAlias = str | int | float | bool
FoldIndex = Annotated[int, Field(ge=0)]


class CategoricalSearchParameter(ConfigModel):
    """Categorical scalar search parameter."""

    path: str = Field(min_length=1)
    """Dot-separated path in the image classification training configuration."""

    type: Literal["categorical"] = "categorical"
    """Optuna distribution type."""

    choices: tuple[SearchChoice, ...] = Field(min_length=2)
    """Values from which Optuna selects one."""

    initial_value: SearchChoice | None = None
    """Optional fixed choice for initial trials; adaptive trials use all choices."""

    @model_validator(mode="after")
    def validate_initial_value(self) -> "CategoricalSearchParameter":
        """Require the initial value to belong to the search distribution."""
        if self.initial_value is not None and self.initial_value not in self.choices:
            raise ValueError("initial_value must belong to choices")
        return self


class ConfigurationChoice(ConfigModel):
    """Named partial training configuration used as one categorical choice."""

    name: str = Field(min_length=1)
    """Value stored in Optuna for this choice."""

    path: Path | None = None
    """YAML overlay applied to the base training configuration, or ``None`` for no change."""


class ConfigurationSearchParameter(ConfigModel):
    """Categorical search over grouped configuration overlays."""

    name: str = Field(min_length=1)
    """Unique Optuna parameter name for this configuration selector."""

    type: Literal["configuration"] = "configuration"
    """Optuna distribution type."""

    choices: tuple[ConfigurationChoice, ...] = Field(min_length=2)
    """Named YAML overlays from which Optuna selects one."""

    @model_validator(mode="after")
    def validate_unique_names(self) -> "ConfigurationSearchParameter":
        """Reject duplicate names that would make a choice ambiguous."""
        names = [choice.name for choice in self.choices]
        if len(names) != len(set(names)):
            raise ValueError("configuration choice names must be unique")
        return self


SearchParameter: TypeAlias = Annotated[
    FloatSearchParameter | IntSearchParameter | CategoricalSearchParameter | ConfigurationSearchParameter,
    Field(discriminator="type"),
]


class StudyConfig(ConfigModel):
    """Optuna study settings."""

    name: str = Field(min_length=1)
    """Persistent study name."""

    storage: Path
    """Journal path on a file system shared by all Slurm jobs."""

    metric: str = "accuracy/validation"
    """Lightning validation metric optimized by the study."""

    direction: Literal["minimize", "maximize"] = "maximize"
    """Desired direction for the study metric."""

    n_trials: int = Field(default=80, ge=1)
    """Number of trials run locally or submitted to Slurm."""

    sampler_seed: int = Field(default=42, ge=0, le=2**32 - 1)
    """Base seed for Optuna's TPE sampler."""

    startup_trials: int = Field(default=10, ge=0)
    """Number of random trials before TPE starts."""

    multivariate: bool = False
    """Sample parameter combinations jointly instead of independently."""

    pruning_startup_trials: int = Field(default=5, ge=0)
    """Number of complete trials required before median pruning starts."""

    pruning_warmup_folds: int = Field(default=2, ge=0)
    """Number of folds each trial completes before it can be pruned."""


class SlurmConfig(ConfigModel):
    """Resources for independent Slurm trial jobs."""

    max_parallel_trials: int = Field(default=8, ge=1)
    """Maximum number of trial jobs that run at the same time."""

    partition: str | None = None
    """Optional Slurm partition."""

    account: str | None = None
    """Optional Slurm account."""

    gres: str | None = "gpu:1"
    """Optional generic resource request."""

    cpus_per_task: int = Field(default=4, ge=1)
    """CPU cores reserved for each trial."""

    memory: str = Field(default="16G", min_length=1)
    """Memory reserved for each trial."""

    time: str = Field(default="8:00:00", min_length=1)
    """Wall-clock limit for each trial."""

    log_dir: Path = Path("runs/training/hpo/slurm")
    """Directory receiving one stdout and stderr file per array task."""

    extra_args: tuple[str, ...] = ()
    """Additional complete sbatch arguments."""


class InitialOptimizerSettings(ConfigModel):
    """AdamW settings shared by all overlay recipes in one initial round."""

    learning_rate: float = Field(gt=0.0, allow_inf_nan=False)
    weight_decay: float = Field(gt=0.0, allow_inf_nan=False)


class HPOConfig(ConfigModel):
    """Top-level configuration for image classification hyperparameter tuning."""

    training_config: Path | None = None
    """Optional training override file; omitted uses Pydantic defaults."""

    clearml_project_name: str | None = None
    """Optional ClearML project override for HPO trial tasks."""

    validation_folds: tuple[FoldIndex, ...] = Field(default=(0, 1, 2), min_length=1)
    """Validation folds evaluated by every trial, in execution order."""

    study: StudyConfig
    """Optuna study and pruning settings."""

    slurm: SlurmConfig = SlurmConfig()
    """Slurm resources and concurrency."""

    search_space: tuple[SearchParameter, ...] = Field(min_length=1)
    """Typed parameters sampled for each trial."""

    initial_optimizer_settings: tuple[InitialOptimizerSettings, ...] = ()
    """Queue every configuration and categorical combination at each optimizer setting."""

    @model_validator(mode="after")
    def validate_unique_parameter_names(self) -> "HPOConfig":
        """Reject duplicate names that would hide sampled values."""
        parameter_names = [
            parameter.name if isinstance(parameter, ConfigurationSearchParameter) else parameter.path for parameter in self.search_space
        ]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("search-space parameter names must be unique")
        if len(self.validation_folds) != len(set(self.validation_folds)):
            raise ValueError("validation_folds must be unique")
        return self


def sample_training_config(
    base_config: dict[str, object],
    search_space: tuple[SearchParameter, ...],
    trial: optuna.Trial | optuna.trial.FixedTrial,
) -> dict[str, object]:
    """Sample a search space and return a validated training configuration.

    Parameters
    ----------
    base_config
        Training settings used when a path is not in the search space.
    search_space
        Typed definitions of the sampled configuration paths.
    trial
        Active Optuna trial.

    Returns
    -------
    dict
        Parsed settings with all sampled values applied.
    """
    parser = create_training_parser(TrainConfig)
    trial_values = parser.parse_object(base_config)
    for configuration in search_space:
        if not isinstance(configuration, ConfigurationSearchParameter):
            continue
        choice_name = trial.suggest_categorical(
            configuration.name,
            tuple(choice.name for choice in configuration.choices),
        )
        choice = next(choice for choice in configuration.choices if choice.name == choice_name)
        if choice.path is not None:
            trial_values = parser.parse_object(load_config_overlay(choice.path), namespace=trial_values)
    optuna_values: dict[str, object] = {}
    for parameter in search_space:
        if isinstance(parameter, ConfigurationSearchParameter):
            continue
        if isinstance(parameter, FloatSearchParameter):
            value = trial.suggest_float(
                parameter.path,
                parameter.low,
                parameter.high,
                step=parameter.step,
                log=parameter.log,
            )
        elif isinstance(parameter, IntSearchParameter):
            value = trial.suggest_int(
                parameter.path,
                parameter.low,
                parameter.high,
                step=parameter.step,
                log=parameter.log,
            )
        else:
            value = cast(SearchChoice, trial.suggest_categorical(parameter.path, parameter.choices))
        optuna_values[parameter.path] = value
    return parser.parse_object(optuna_values, namespace=trial_values).as_dict()


def load_config_overlay(path: Path) -> dict[str, object]:
    """Load one partial training configuration from YAML."""
    try:
        loaded_values = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot load configuration overlay {path}") from error
    if not isinstance(loaded_values, dict):
        raise ValueError(f"Configuration overlay {path} must contain a mapping")
    return cast(dict[str, object], loaded_values)


def load_base_training_config(config: HPOConfig) -> dict[str, object]:
    """Parse base settings without constructing a network."""
    parser = create_training_parser(TrainConfig)
    arguments = [] if config.training_config is None else ["--config", str(config.training_config)]
    parsed_values = parser.parse_args(arguments)
    return cast(dict[str, object], json.loads(parser.dump(parsed_values, format="json")))


def create_storage(path: Path) -> JournalStorage:
    """Create the Optuna journal storage and its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return JournalStorage(JournalFileBackend(str(path)))


class FoldMedianPruner(optuna.pruners.BasePruner):
    """Compare cumulative fold scores at the same fold count."""

    def __init__(self, startup_trials: int, warmup_folds: int) -> None:
        """Set the minimum evidence required for pruning.

        Parameters
        ----------
        startup_trials
            Completed trials required before comparing scores.
        warmup_folds
            Folds the current trial must finish before comparison.
        """
        self.startup_trials = startup_trials
        self.warmup_folds = warmup_folds

    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        """Prune when the latest fold mean is worse than the reference median."""
        step = trial.last_step
        if step is None or step + 1 < self.warmup_folds:
            return False
        completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if len(completed) < self.startup_trials:
            return False
        reference = [
            other.intermediate_values[step]
            for other in completed
            if step in other.intermediate_values and np.isfinite(other.intermediate_values[step])
        ]
        if not reference:
            return False
        current = trial.intermediate_values[step]
        if not np.isfinite(current):
            return True
        if study.direction == optuna.study.StudyDirection.MAXIMIZE:
            return current < median(reference)
        return current > median(reference)


def create_study(config: HPOConfig) -> optuna.Study:
    """Create or resume the configured study for this worker process."""
    worker_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    sampler = optuna.samplers.TPESampler(
        seed=config.study.sampler_seed + worker_index,
        n_startup_trials=config.study.startup_trials,
        constant_liar=True,
        multivariate=config.study.multivariate,
    )
    pruner = FoldMedianPruner(
        startup_trials=config.study.pruning_startup_trials,
        warmup_folds=config.study.pruning_warmup_folds,
    )
    return optuna.create_study(
        study_name=config.study.name,
        storage=create_storage(config.study.storage),
        sampler=sampler,
        pruner=pruner,
        direction=config.study.direction,
        load_if_exists=True,
    )


def enqueue_initial_trials(study: optuna.Study, config: HPOConfig) -> None:
    """Seed an empty study with matched configuration and categorical choices.

    Parameters
    ----------
    study
        Shared study. Existing trials, including waiting trials, prevent reseeding.
    config
        Configuration selectors, categorical choices, and optimizer settings.

    Notes
    -----
    Call only from the submission process, before workers start. Queued trials
    count toward the submission budget; remaining trials use the sampler.
    """
    if not config.initial_optimizer_settings or study.get_trials(deepcopy=False):
        return
    choices_by_name: dict[str, tuple[SearchChoice, ...]] = {}
    for parameter in config.search_space:
        if isinstance(parameter, ConfigurationSearchParameter):
            choices_by_name[parameter.name] = tuple(choice.name for choice in parameter.choices)
        elif isinstance(parameter, CategoricalSearchParameter):
            choices_by_name[parameter.path] = parameter.choices if parameter.initial_value is None else (parameter.initial_value,)
    parameters = {parameter.path: parameter for parameter in config.search_space if isinstance(parameter, FloatSearchParameter)}
    queued: list[dict[str, SearchChoice]] = []
    for settings in config.initial_optimizer_settings:
        optimizer = {"training.learning_rate": settings.learning_rate, "training.weight_decay": settings.weight_decay}
        for name, value in optimizer.items():
            parameter = parameters.get(name)
            if parameter is None or parameter.step is not None or not parameter.low <= value <= parameter.high:
                raise ValueError(f"Initial {name} must lie in a continuous float search range")
        for choices in product(*choices_by_name.values()):
            queued.append({**dict(zip(choices_by_name, choices, strict=True)), **optimizer})
    if len(queued) > config.study.n_trials:
        raise ValueError("Initial trial count exceeds the submission budget")
    for index in np.random.default_rng(config.study.sampler_seed).permutation(len(queued)):
        study.enqueue_trial(queued[index])
    logger.info("Queued %d initial trials within the %d-trial budget", len(queued), config.study.n_trials)


def train_fold(
    fold_values: Namespace,
    run_dir: Path,
    metric: str,
    mode: Literal["min", "max"],
) -> float:
    """Train one validation fold and return its best monitored score.

    Parameters
    ----------
    fold_values
        Parsed settings with the network, seed and validation index.
    run_dir
        Directory for this fold's logs and checkpoint.
    metric
        Validation metric used for stopping and checkpoint selection.
    mode
        Whether lower or higher metric values are better.

    Returns
    -------
    float
        Best monitored checkpoint score.

    Raises
    ------
    RuntimeError
        If training produces no monitored checkpoint score.
    """
    parser = create_training_parser(TrainConfig)
    pl.seed_everything(fold_values.seed, workers=True)
    fold_config, network = instantiate_training_config(parser, TrainConfig, fold_values)
    resolved_fold_config = dump_training_config(parser, fold_values, fold_config)
    if fold_config.resume_from is not None:
        raise ValueError("HPO trials must start fresh; resume_from must be null")
    data_module = ImageDataModule(fold_config.dataset)
    data_module.setup("fit")
    module = ClassificationModule(
        fold_config.training,
        network=network,
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=run_dir,
        name="tensorboard",
        prefix=f"fold_{fold_config.dataset.validation_fold}",
    )
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best",
        monitor=metric,
        mode=mode,
        save_top_k=1,
    )
    trainer = pl.Trainer(
        max_epochs=fold_config.max_epochs,
        accelerator=fold_config.accelerator,
        devices=fold_config.devices,
        precision=fold_config.precision,
        default_root_dir=run_dir,
        logger=tensorboard_logger,
        callbacks=[
            EarlyStopping(
                monitor=metric,
                mode=mode,
                patience=fold_config.early_stopping_patience,
            ),
            checkpoint,
            ConfigCheckpoint(resolved_fold_config),
        ],
    )
    trainer.fit(module, datamodule=data_module)
    if checkpoint.best_model_score is None:
        raise RuntimeError(f"Metric {metric!r} is not available after fold {fold_config.dataset.validation_fold}")

    return float(checkpoint.best_model_score.item())


def create_objective(
    config: HPOConfig,
    base_config: dict[str, object],
) -> Callable[[optuna.Trial], float]:
    """Create the image classification training objective for one study worker."""
    # Avoid the SDK's subprocess startup race, which can hang Task.close().
    clearml_config.get("development")["report_use_subprocess"] = False

    def objective(trial: optuna.Trial) -> float:
        trial_values = sample_training_config(
            base_config,
            config.search_space,
            trial,
        )
        parser = create_training_parser(TrainConfig)
        parsed_trial_values = parser.parse_object(trial_values)
        invalid_folds = [fold for fold in config.validation_folds if fold >= parsed_trial_values.dataset.num_folds]
        if invalid_folds:
            raise ValueError(f"validation_folds {invalid_folds} must be smaller than dataset.num_folds={parsed_trial_values.dataset.num_folds}")
        start_time = datetime.now().strftime("%Y%m%d%H%M%S")
        random_name = generate_slug(2).replace("-", "_")
        run_id = f"training_{start_time}_HPO_{trial.number:04d}_{random_name}"
        task_tags = ["hpo", config.study.name, f"trial_{trial.number}"]
        for parameter in config.search_space:
            if isinstance(parameter, ConfigurationSearchParameter):
                task_tags.append(f"{parameter.name}_{trial.params[parameter.name]}")
        task = None
        project = config.clearml_project_name or parsed_trial_values.clearml_project_name
        if project is not None:
            Task.set_random_seed(None)
            task = Task.init(project_name=project, task_name=run_id, tags=task_tags, reuse_last_task_id=False)
        try:
            resolved_trial_config = json.loads(parser.dump(parsed_trial_values, format="json"))
            if task is not None:
                task.connect_configuration(resolved_trial_config, name="training", ignore_remote_overrides=True)
                task.connect(
                    {
                        "study": config.study.name,
                        "trial": trial.number,
                        "parameters": trial.params,
                        "validation_folds": list(config.validation_folds),
                    },
                    name="optuna",
                )
            metric_mode = "max" if config.study.direction == "maximize" else "min"
            fold_scores: list[float] = []
            clearml_logger = task.get_logger() if task is not None else None
            logger.info("Trial %d starting validation folds %s: %s", trial.number, config.validation_folds, trial.params)
            for fold_step, validation_fold in enumerate(config.validation_folds):
                logger.info(
                    "Trial %d training validation fold %d (%d/%d)",
                    trial.number,
                    validation_fold,
                    fold_step + 1,
                    len(config.validation_folds),
                )
                fold_seed = (parsed_trial_values.seed + validation_fold) % (2**32)
                fold_run_dir = parsed_trial_values.output_dir / run_id / f"fold_{validation_fold}"
                parsed_fold_values = parser.parse_object(
                    {"seed": fold_seed, "dataset": {"validation_fold": validation_fold}},
                    namespace=parsed_trial_values.clone(),
                )
                fold_score = train_fold(parsed_fold_values, fold_run_dir, config.study.metric, metric_mode)
                fold_scores.append(fold_score)
                running_mean = fmean(fold_scores)
                logger.info(
                    "Trial %d fold %d %s=%g, running mean=%g",
                    trial.number,
                    validation_fold,
                    config.study.metric,
                    fold_score,
                    running_mean,
                )
                trial.set_user_attr(
                    "fold_scores",
                    dict(zip(map(str, config.validation_folds[: len(fold_scores)]), fold_scores, strict=True)),
                )
                trial.report(running_mean, step=fold_step)
                if clearml_logger is not None:
                    clearml_logger.report_scalar(
                        title=f"HPO fold {config.study.metric}",
                        series=f"fold_{validation_fold}",
                        value=fold_score,
                        iteration=0,
                    )
                    clearml_logger.report_scalar(
                        title=f"HPO {config.study.metric}",
                        series="running_mean",
                        value=running_mean,
                        iteration=fold_step,
                    )

                if fold_step < len(config.validation_folds) - 1 and trial.should_prune():
                    raise optuna.TrialPruned(f"Pruned after validation fold {validation_fold}")

            mean_score = fmean(fold_scores)
            score_standard_deviation = pstdev(fold_scores)
            if clearml_logger is not None:
                clearml_logger.report_single_value(name=f"mean {config.study.metric}", value=mean_score)
                clearml_logger.report_single_value(name=f"standard deviation {config.study.metric}", value=score_standard_deviation)
            trial.set_user_attr("fold_score_standard_deviation", score_standard_deviation)
            logger.info(
                "Trial %d validation folds %s mean %s=%g, standard deviation=%g",
                trial.number,
                config.validation_folds,
                config.study.metric,
                mean_score,
                score_standard_deviation,
            )
            return mean_score
        finally:
            if task is not None:
                task.close()

    return objective


def run_hpo(config: HPOConfig) -> optuna.Study:
    """Run the configured number of trials in the current process."""
    base_config = load_base_training_config(config)
    study = create_study(config)
    study.optimize(create_objective(config, base_config), n_trials=config.study.n_trials)
    complete_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    if complete_trials:
        logger.info("Best trial %d: %s=%g", study.best_trial.number, config.study.metric, study.best_value)
        logger.info("Best parameters: %s", study.best_params)
    else:
        logger.info("The study has no complete trials yet")
    return study


def build_sbatch_command(config: HPOConfig, config_path: Path) -> list[str]:
    """Build one Slurm array command with one Optuna trial per task."""
    slurm = config.slurm
    array = f"0-{config.study.n_trials - 1}%{slurm.max_parallel_trials}"
    worker_command = shlex.join(
        [
            sys.executable,
            "-m",
            "computer_vision.hpo",
            "--config",
            str(config_path.resolve()),
            "--worker",
        ]
    )
    command = [
        "sbatch",
        f"--job-name={config.study.name}",
        f"--array={array}",
        f"--cpus-per-task={slurm.cpus_per_task}",
        f"--mem={slurm.memory}",
        f"--time={slurm.time}",
        f"--chdir={Path.cwd()}",
        f"--output={slurm.log_dir.resolve()}/%A_%a.out",
        f"--error={slurm.log_dir.resolve()}/%A_%a.err",
    ]
    if slurm.partition is not None:
        command.append(f"--partition={slurm.partition}")
    if slurm.account is not None:
        command.append(f"--account={slurm.account}")
    if slurm.gres is not None:
        command.append(f"--gres={slurm.gres}")
    command.extend(slurm.extra_args)
    command.append(f"--wrap={worker_command}")
    return command


def parse_arguments() -> tuple[Path, bool, bool]:
    """Parse the public submission command or internal worker command."""
    parser = argparse.ArgumentParser(description="Submit one Slurm array task per image classification Optuna trial.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--local", action="store_true", help="Run trials locally instead of submitting to Slurm.")
    arguments = parser.parse_args()
    return arguments.config, arguments.worker, arguments.local


def main() -> None:
    """Submit HPO trials to Slurm or run one internal array worker."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path, worker, local = parse_arguments()
    config = parse_and_validate(HPOConfig, ["--config", str(config_path)])
    if worker:
        worker_study = config.study.model_copy(update={"n_trials": 1})
        run_hpo(config.model_copy(update={"study": worker_study}))
        return
    study = create_study(config)
    enqueue_initial_trials(study, config)
    if local:
        run_hpo(config)
        return
    config.slurm.log_dir.mkdir(parents=True, exist_ok=True)
    command = build_sbatch_command(config, config_path)
    logger.info("Submitting %d trials with at most %d running", config.study.n_trials, config.slurm.max_parallel_trials)
    subprocess.run(command, check=True)  # noqa: S603 - arguments remain separate and shell expansion is disabled.


if __name__ == "__main__":
    main()
