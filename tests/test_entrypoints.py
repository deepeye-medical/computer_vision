"""Classification command-line tests."""

import sys
from pathlib import Path
from unittest.mock import Mock, call

import pytest
import torch

import computer_vision.train as train


@pytest.mark.parametrize("outcome", ["complete", "fit_error", "connection_error"])
def test_train_seeds_network_initialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    """Seed network construction and close tracking on setup or training failure."""
    states = []
    for ambient_seed in (101, 202):
        torch.manual_seed(ambient_seed)
        task_api = Mock()
        task = task_api.init.return_value
        development = {"report_use_subprocess": True}
        sdk_config = Mock()
        sdk_config.get.return_value = development
        monkeypatch.setattr(train, "clearml_config", sdk_config)

        trainer = Mock()
        if outcome == "fit_error":
            trainer.fit.side_effect = RuntimeError("synthetic fit failure")
        if outcome == "connection_error":
            task.connect_configuration.side_effect = RuntimeError("synthetic connection failure")
        monkeypatch.setattr(train, "Task", task_api)
        monkeypatch.setattr(train.pl, "Trainer", Mock(return_value=trainer))
        data_module = Mock()
        monkeypatch.setattr(train, "ImageDataModule", data_module)
        monkeypatch.setattr(train, "TensorBoardLogger", Mock())
        monkeypatch.setattr(sys, "argv", ["computer-vision-train", "--output_dir", str(tmp_path), "--clearml_project_name", "test"])
        if outcome == "complete":
            train.main()
        else:
            with pytest.raises(RuntimeError, match="synthetic"):
                train.main()
        assert task_api.mock_calls[0] == call.set_random_seed(None)
        assert task_api.mock_calls[1][0] == "init"
        assert development["report_use_subprocess"] is False
        task.close.assert_called_once_with()
        if outcome != "connection_error":
            assert torch.initial_seed() == 42
            module = trainer.fit.call_args.args[0]
            states.append({name: value.clone() for name, value in module.network.state_dict().items()})
    if states:
        for name, value in states[0].items():
            torch.testing.assert_close(states[1][name], value, rtol=0, atol=0)
