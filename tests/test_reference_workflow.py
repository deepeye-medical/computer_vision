"""Check synthetic image preparation, training, and viewing."""

import json
from pathlib import Path

import pytest
import torch

from computer_vision.config import parse_and_validate
from computer_vision.data import DatasetConfig, ImageDataModule, ManifestDataset
from computer_vision.data_analysis.report import ReportConfig, report
from computer_vision.data_analysis.viewer import ViewerConfig, load_viewer_checkpoint, render_page
from computer_vision.data_preparation.dummy import DummyConfig, generate
from computer_vision.data_preparation.integration import IntegrationConfig, integrate
from computer_vision.data_preparation.source_a.prepare import SourceConfig
from computer_vision.data_preparation.source_a.prepare import prepare as prepare_a
from computer_vision.data_preparation.source_b.prepare import prepare as prepare_b
from computer_vision.hpo import HPOConfig, create_study, run_hpo
from computer_vision.lightning_module import ClassificationModule
from computer_vision.manifest import read_manifest, write_manifest
from computer_vision.train import TrainConfig, create_training_parser, dump_training_config, instantiate_training_config


@pytest.fixture
def integrated_manifest(tmp_path: Path) -> Path:
    """Prepare both layouts and combine their portable records."""
    source = tmp_path / "inputs"
    generate(DummyConfig(output=source))
    manifests = []
    for name, prepare in (("source_a", prepare_a), ("source_b", prepare_b)):
        manifest = tmp_path / "prepared" / f"{name}.csv"
        prepare(SourceConfig(source=source / name, source_name=name, output=manifest))
        manifests.append(manifest)
    output = tmp_path / "integrated" / "images.csv"
    integrate(IntegrationConfig(manifests=tuple(manifests), output=output))
    return output


def test_connected_reference(integrated_manifest: Path) -> None:
    """Share class indices, paths, transforms, and stable fold membership."""
    config = DatasetConfig(kind="manifest", root=integrated_manifest)
    module = ImageDataModule(config)
    module.setup()
    assert len(module.train_dataset) + len(module.validation_dataset) == 20
    assert isinstance(module.test_dataset, ManifestDataset)
    assert isinstance(module.train_dataset.dataset, ManifestDataset)
    assert len(module.test_dataset) == 4
    assert not set(module.train_dataset.indices) & set(module.validation_dataset.indices)
    dataset = ManifestDataset(config)
    image, label = dataset[0]
    assert image.shape == (3, 32, 32)
    assert 0 <= image.min() <= image.max() <= 1
    assert label == dataset.class_to_idx[dataset.records[0].label]
    assert module.test_dataset.class_to_idx == dataset.class_to_idx
    viewer = ViewerConfig(dataset=config)
    rng_state = torch.get_rng_state()
    page = render_page(viewer, {"source": ["source_a"], "label": ["circle"]})
    selected_count = sum(
        dataset.records[index].source == "source_a" and dataset.records[index].label == "circle" for index in module.train_dataset.indices
    )
    assert f"Sample 1 of {selected_count}" in page
    assert "Model input" in page and "Augmentation preview" in page
    assert "data:image/png;base64," in page
    validation_page = render_page(viewer, {"split": ["validation"]})
    assert f"Sample 1 of {len(module.validation_dataset)}" in validation_page
    # Only augmentation previews may consume randomness, and those restore it.
    torch.testing.assert_close(rng_state, torch.get_rng_state())
    report_path = integrated_manifest.with_suffix(".json")
    report(ReportConfig(manifest=integrated_manifest, output=report_path))
    summary = json.loads(report_path.read_text())
    assert summary["samples"] == 24
    assert summary["classes"] == {"circle": 12, "square": 12}
    assert summary["duplicate_images"] == summary["missing_images"] == summary["missing_labels"] == 0
    assert summary["train_test_duplicate_images"] == 0
    assert summary["groups_by_split"] == {"train": 12, "test": 4}
    fingerprint = module.training_fingerprint()
    records = read_manifest(integrated_manifest)
    selected_id = module.train_dataset.dataset.records[module.train_dataset.indices[0]].sample_id
    write_manifest(
        integrated_manifest,
        [
            record.model_copy(update={"label": "square" if record.label == "circle" else "circle"}) if record.sample_id == selected_id else record
            for record in records
        ],
    )
    changed = ImageDataModule(config)
    changed.setup("fit")
    assert changed.training_fingerprint() != fingerprint


def test_integration_rejects_overlap(integrated_manifest: Path) -> None:
    """Reject repeated source inputs before they reach training."""
    with pytest.raises(ValueError, match="unique"):
        integrate(IntegrationConfig(manifests=(integrated_manifest, integrated_manifest), output=integrated_manifest.with_name("duplicate.csv")))


def test_group_folds(integrated_manifest: Path) -> None:
    """Keep groups intact, cover each group once, and ignore row order and seeds."""
    from computer_vision.data import assign_group_folds

    records = read_manifest(integrated_manifest)
    groups = [record.group_id for record in records if record.split == "train"]
    expected = dict(zip(groups, assign_group_folds(groups, 3), strict=True))
    reversed_groups = list(reversed(groups))
    assert dict(zip(reversed_groups, assign_group_folds(reversed_groups, 3), strict=True)) == expected
    validation_groups = []
    for fold in range(3):
        module = ImageDataModule(DatasetConfig(kind="manifest", root=integrated_manifest, validation_fold=fold, split_seed=fold))
        module.setup("fit")
        train_groups = {groups[index] for index in module.train_dataset.indices}
        held_out = {groups[index] for index in module.validation_dataset.indices}
        assert not train_groups & held_out
        validation_groups.extend(held_out)
    assert sorted(validation_groups) == sorted(set(groups))
    train_group = groups[0]
    write_manifest(
        integrated_manifest, [record.model_copy(update={"group_id": train_group}) if record.split == "test" else record for record in records]
    )
    with pytest.raises(ValueError, match="overlap"):
        ManifestDataset(DatasetConfig(kind="manifest", root=integrated_manifest))


def test_manifest_hpo(integrated_manifest: Path) -> None:
    """Run the connected HPO recipe on grouped images with full augmentation."""
    torch.set_num_threads(1)
    output = integrated_manifest.parent
    training_path = output / "training.yaml"
    training_path.write_text(
        Path("configs/train_manifest.yaml")
        .read_text()
        .replace(".artifacts/images.csv", str(integrated_manifest))
        .replace("max_epochs: 3", "max_epochs: 1")
        + f"\noutput_dir: {output / 'runs'}\n"
    )
    config = parse_and_validate(HPOConfig, ["--config", "configs/hpo_manifest.yaml"])
    config = config.model_copy(
        update={
            "training_config": training_path,
            "study": config.study.model_copy(update={"storage": output / "study.log", "n_trials": 1}),
        }
    )
    study = create_study(config)
    study.enqueue_trial({"augmentation_recipe": "mild", "loss_recipe": "bce", "training.learning_rate": 0.001})
    study = run_hpo(config)
    assert len(study.trials) == 1
    assert 0 <= study.best_value <= 1
    assert len(list((output / "runs").rglob("best.ckpt"))) == 3


def test_viewer_checkpoint(integrated_manifest: Path) -> None:
    """Rebuild CPU inference with saved preprocessing and named softmax outputs."""
    parser = create_training_parser(TrainConfig)
    parsed = parser.parse_args(
        [
            "--config",
            "configs/train_manifest.yaml",
            "--dataset.root",
            str(integrated_manifest),
            "--dataset.image_size",
            "16",
            "--dataset.validation_fold",
            "1",
        ]
    )
    training_config, network = instantiate_training_config(parser, TrainConfig, parsed)
    module = ClassificationModule(training_config.training, network).eval()
    checkpoint = integrated_manifest.with_suffix(".ckpt")
    torch.save(
        {
            "run_config": dump_training_config(parser, parsed, training_config),
            "state_dict": module.state_dict(),
            "class_names": ["circle", "square"],
        },
        checkpoint,
    )
    viewer, model = load_viewer_checkpoint(
        ViewerConfig(
            dataset=DatasetConfig(kind="manifest", root=integrated_manifest, image_size=64),
            checkpoint=checkpoint,
        )
    )
    assert model is not None and not model.training
    assert next(model.parameters()).device.type == "cpu"
    assert viewer.dataset.image_size == 16 and viewer.dataset.validation_fold == 1
    dataset = ManifestDataset(viewer.dataset)
    splits = ImageDataModule(viewer.dataset)
    splits.setup("fit")
    model_input, _ = dataset[splits.train_dataset.indices[0]]
    with torch.inference_mode():
        expected_scores = module(model_input.unsqueeze(0)).softmax(dim=1)[0]
    page = render_page(viewer, {}, model)
    assert "CPU prediction" in page and "Softmax score" in page
    for name, score in zip(dataset.classes, expected_scores, strict=True):
        assert f"<td>{name}</td><td>{float(score):.6f}</td>" in page
    assert page == render_page(viewer, {}, model)
    legacy = torch.load(checkpoint, weights_only=True)
    legacy.pop("class_names")
    torch.save(legacy, checkpoint)
    viewer, legacy_model = load_viewer_checkpoint(viewer)
    legacy_page = render_page(viewer, {}, legacy_model)
    assert "Class 0" in legacy_page and "Scores use output indices" in legacy_page
    model.class_names = ("different", "labels")
    with pytest.raises(ValueError, match="class ordering"):
        render_page(viewer, {}, model)
