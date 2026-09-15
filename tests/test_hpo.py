"""HPO objective and Slurm command tests."""

import shlex
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Literal
from unittest.mock import Mock, call

import optuna
import pytest

import computer_vision.hpo as hpo
from computer_vision.config import parse_and_validate


def test_hpo_reporting_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure reporting threads before creating a ClearML task."""
    config = parse_and_validate(hpo.HPOConfig, ["--config", "configs/hpo.yaml"])
    development = {"report_use_subprocess": True}
    sdk_config = Mock()
    sdk_config.get.return_value = development
    monkeypatch.setattr(hpo, "clearml_config", sdk_config)

    hpo.create_objective(config, {})

    sdk_config.get.assert_called_once_with("development")
    assert development["report_use_subprocess"] is False


@pytest.mark.parametrize("outcome", ["complete", "pruned", "connection_error"])
def test_hpo_objective(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    """Check median pruning, fold settings, and tracking cleanup."""
    config = parse_and_validate(hpo.HPOConfig, ["--config", "configs/hpo.yaml"])
    config = config.model_copy(update={"validation_folds": (2, 0, 1), "clearml_project_name": "test"})
    instantiate = Mock(side_effect=AssertionError("Configuration-only HPO must not construct a network"))
    monkeypatch.setattr(hpo, "instantiate_training_config", instantiate)
    base = hpo.load_base_training_config(config)
    base["output_dir"] = str(tmp_path)
    original_base = deepcopy(base)
    task_api = Mock()
    task = task_api.init.return_value
    if outcome == "connection_error":
        task.connect_configuration.side_effect = RuntimeError("synthetic connection failure")
    monkeypatch.setattr(hpo, "Task", task_api)
    fold = Mock(return_value=0.1 if outcome == "pruned" else 0.9)
    monkeypatch.setattr(hpo, "train_fold", fold)
    monkeypatch.setattr(hpo, "create_storage", lambda _path: optuna.storages.InMemoryStorage())
    study = hpo.create_study(config)
    for _ in range(config.study.pruning_startup_trials):
        study.add_trial(optuna.trial.create_trial(value=0.8, intermediate_values={0: 0.8, 1: 0.8, 2: 0.8}))
    study.optimize(hpo.create_objective(config, base), n_trials=1, catch=(RuntimeError,))
    trial = study.trials[-1]
    instantiate.assert_not_called()
    assert base == original_base
    task_api.set_random_seed.assert_called_once_with(None)
    assert task_api.mock_calls[0] == call.set_random_seed(None)
    assert task_api.mock_calls[1][0] == "init"
    task.close.assert_called_once_with()
    expected_count = {"complete": 3, "pruned": 2, "connection_error": 0}[outcome]
    assert fold.call_count == expected_count
    assert [entry.args[0].dataset.validation_fold for entry in fold.call_args_list] == list(config.validation_folds[:expected_count])
    assert [entry.args[0].seed for entry in fold.call_args_list] == [44, 42, 43][:expected_count]
    states = {
        "complete": optuna.trial.TrialState.COMPLETE,
        "pruned": optuna.trial.TrialState.PRUNED,
        "connection_error": optuna.trial.TrialState.FAIL,
    }
    assert trial.state == states[outcome]
    if expected_count:
        score = 0.1 if outcome == "pruned" else 0.9
        assert trial.user_attrs["fold_scores"] == {str(index): score for index in config.validation_folds[:expected_count]}
        for entry in fold.call_args_list:
            assert entry.args[1].parent.parent == tmp_path
            assert entry.args[2:] == (config.study.metric, "max")
            assert entry.args[0].network.class_path == "computer_vision.model.ResNet"
            assert entry.args[0].network.init_args.dropout == trial.params["network.init_args.dropout"]
    if outcome == "complete":
        assert trial.value == pytest.approx(0.9)
        assert trial.user_attrs["fold_score_standard_deviation"] == 0.0


def test_build_sbatch_command(tmp_path: Path) -> None:
    """Preserve a config path containing spaces in the worker command."""
    config = parse_and_validate(hpo.HPOConfig, ["--config", "configs/hpo.yaml"])
    config_path = tmp_path / "config with spaces.yaml"
    command = hpo.build_sbatch_command(config, config_path)
    assert command[0] == "sbatch"
    assert f"--array=0-{config.study.n_trials - 1}%{config.slurm.max_parallel_trials}" in command
    wrapped = next(argument.removeprefix("--wrap=") for argument in command if argument.startswith("--wrap="))
    worker = shlex.split(wrapped)
    assert worker[1:4] == ["-m", "computer_vision.hpo", "--config"]
    assert worker[4:] == [str(config_path), "--worker"]


@pytest.mark.parametrize("initial_dropout", [None, 0.2])
def test_initial_trial_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, initial_dropout: float | None) -> None:
    """Queue matched combinations once, in a reproducible order, within budget."""
    config = hpo.HPOConfig(
        study=hpo.StudyConfig(name="coverage", storage=tmp_path / "study.log", n_trials=20, multivariate=True),
        search_space=(
            hpo.ConfigurationSearchParameter(
                name="input", choices=(hpo.ConfigurationChoice(name="binary"), hpo.ConfigurationChoice(name="probability"))
            ),
            hpo.ConfigurationSearchParameter(name="model", choices=(hpo.ConfigurationChoice(name="a"), hpo.ConfigurationChoice(name="b"))),
            hpo.CategoricalSearchParameter(path="network.init_args.dropout", choices=(0.0, 0.2), initial_value=initial_dropout),
            hpo.FloatSearchParameter(path="training.learning_rate", low=1e-5, high=1e-3, log=True),
            hpo.FloatSearchParameter(path="training.weight_decay", low=1e-6, high=1e-2, log=True),
        ),
        initial_optimizer_settings=(
            hpo.InitialOptimizerSettings(learning_rate=1e-3, weight_decay=1e-4),
            hpo.InitialOptimizerSettings(learning_rate=1e-4, weight_decay=1e-3),
        ),
    )
    monkeypatch.setattr(hpo, "create_storage", lambda _path: optuna.storages.InMemoryStorage())
    study = hpo.create_study(config)
    hpo.enqueue_initial_trials(study, config)
    queued = [trial.system_attrs["fixed_params"] for trial in study.trials]
    assert len(queued) == (16 if initial_dropout is None else 8)
    assert Counter((trial["input"], trial["model"], trial["network.init_args.dropout"]) for trial in queued) == {
        (input_name, model_name, dropout): 2
        for input_name in ("binary", "probability")
        for model_name in ("a", "b")
        for dropout in ((0.0, 0.2) if initial_dropout is None else (initial_dropout,))
    }
    per_anchor = len(queued) // 2
    assert Counter((trial["training.learning_rate"], trial["training.weight_decay"]) for trial in queued) == {
        (1e-3, 1e-4): per_anchor,
        (1e-4, 1e-3): per_anchor,
    }
    hpo.enqueue_initial_trials(study, config)
    assert len(study.trials) == len(queued)
    repeated = hpo.create_study(config)
    hpo.enqueue_initial_trials(repeated, config)
    assert queued == [trial.system_attrs["fixed_params"] for trial in repeated.trials]


@pytest.mark.parametrize("direction,sign", [("maximize", 1), ("minimize", -1)])
def test_fold_pruner_uses_latest_mean(direction: Literal["maximize", "minimize"], sign: int) -> None:
    """An easy first fold must not shield a weak two-fold mean."""
    study = optuna.create_study(direction=direction, pruner=hpo.FoldMedianPruner(startup_trials=2, warmup_folds=2))
    trial = study.ask()
    trial.report(sign * 0.79, step=0)
    assert not trial.should_prune()
    trial.report(sign * 0.74, step=1)
    study.add_trial(optuna.trial.create_trial(value=sign * 0.75, intermediate_values={0: sign * 0.80, 1: sign * 0.75}))
    assert not trial.should_prune()
    study.add_trial(optuna.trial.create_trial(value=sign * 0.75, intermediate_values={0: sign * 0.80, 1: sign * 0.75}))
    assert trial.should_prune()
    better = study.ask()
    better.report(sign * 0.81, step=0)
    assert not better.should_prune()
    better.report(sign * 0.76, step=1)
    assert not better.should_prune()
