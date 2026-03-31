from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config.types import AugmentationConfig, ExperimentConfig
from src.engine.trainer import Trainer
from src.models.early_exit_model import EarlyExitModel
from src.models.mlp_backbone import MLPBackbone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_CLASSES = 10
HIDDEN_DIM = 32
NUM_LAYERS = 3
EXIT_INDICES = [1, 2]
# one weight per exit head + final head
LOSS_WEIGHTS = [1.0 / (len(EXIT_INDICES) + 1)] * (len(EXIT_INDICES) + 1)


def make_config(
    optimizer: str = "sgd",
    lr_scheduler: str | None = None,
    output_dir: str = "/tmp/trainer_test",
) -> ExperimentConfig:
    return ExperimentConfig(
        backbone="mlp",
        exit_layer_indices=EXIT_INDICES,
        confidence_method="max_softmax",
        exit_loss_weights=LOSS_WEIGHTS,
        optimizer=optimizer,
        learning_rate=1e-3,
        lr_scheduler=lr_scheduler,
        num_epochs=2,
        batch_size=4,
        threshold_sweep=[0.5, 0.9],
        dataset="cifar10",
        dataset_path="/tmp/data",
        augmentation=AugmentationConfig(),
        random_seed=42,
        output_dir=output_dir,
    )


def make_model() -> EarlyExitModel:
    backbone = MLPBackbone(num_layers=NUM_LAYERS, num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM)
    return EarlyExitModel(
        backbone=backbone,
        exit_layer_indices=EXIT_INDICES,
        num_classes=NUM_CLASSES,
    )


def make_loader(batch_size: int = 4, num_samples: int = 8) -> DataLoader:
    inputs = torch.randn(num_samples, 3, 32, 32)
    targets = torch.randint(0, NUM_CLASSES, (num_samples,))
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)


# ---------------------------------------------------------------------------
# Optimizer tests
# ---------------------------------------------------------------------------

def test_sgd_trains_one_step_without_error(tmp_path):
    config = make_config(optimizer="sgd", output_dir=str(tmp_path))
    model = make_model()
    trainer = Trainer(model, config)
    loader = make_loader()
    loss = trainer.train_epoch(loader)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_adam_trains_one_step_without_error(tmp_path):
    config = make_config(optimizer="adam", output_dir=str(tmp_path))
    model = make_model()
    trainer = Trainer(model, config)
    loader = make_loader()
    loss = trainer.train_epoch(loader)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_invalid_optimizer_raises(tmp_path):
    config = make_config(optimizer="rmsprop", output_dir=str(tmp_path))
    model = make_model()
    with pytest.raises(ValueError, match="rmsprop"):
        Trainer(model, config)


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------

def test_cosine_scheduler_steps_correctly(tmp_path):
    config = make_config(optimizer="sgd", lr_scheduler="cosine", output_dir=str(tmp_path))
    model = make_model()
    trainer = Trainer(model, config)
    initial_lr = trainer.optimizer.param_groups[0]["lr"]
    trainer.scheduler.step()
    stepped_lr = trainer.optimizer.param_groups[0]["lr"]
    # After one cosine step the LR should change (decrease from initial).
    assert stepped_lr != initial_lr or config.num_epochs == 1


def test_step_scheduler_steps_correctly(tmp_path):
    config = make_config(optimizer="sgd", lr_scheduler="step", output_dir=str(tmp_path))
    model = make_model()
    trainer = Trainer(model, config)
    initial_lr = trainer.optimizer.param_groups[0]["lr"]
    # StepLR with step_size=30 should not change LR before 30 steps.
    for _ in range(29):
        trainer.scheduler.step()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(initial_lr)
    trainer.scheduler.step()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(initial_lr * 0.1)


def test_no_scheduler_is_none(tmp_path):
    config = make_config(optimizer="sgd", lr_scheduler=None, output_dir=str(tmp_path))
    model = make_model()
    trainer = Trainer(model, config)
    assert trainer.scheduler is None


def test_invalid_scheduler_raises(tmp_path):
    config = make_config(optimizer="sgd", lr_scheduler="plateau", output_dir=str(tmp_path))
    model = make_model()
    with pytest.raises(ValueError, match="plateau"):
        Trainer(model, config)


# ---------------------------------------------------------------------------
# Config log file tests (Req 8.2)
# ---------------------------------------------------------------------------

def test_config_yaml_written_to_output_dir(tmp_path):
    config = make_config(output_dir=str(tmp_path))
    model = make_model()
    Trainer(model, config)
    config_file = tmp_path / "config.yaml"
    assert config_file.exists()


def test_config_yaml_contains_optimizer_field(tmp_path):
    config = make_config(optimizer="adam", output_dir=str(tmp_path))
    model = make_model()
    Trainer(model, config)
    content = (tmp_path / "config.yaml").read_text()
    assert "adam" in content


def test_config_yaml_creates_output_dir_if_missing(tmp_path):
    nested = tmp_path / "nested" / "subdir"
    config = make_config(output_dir=str(nested))
    model = make_model()
    Trainer(model, config)
    assert (nested / "config.yaml").exists()


# ---------------------------------------------------------------------------
# Weighted loss tests
# ---------------------------------------------------------------------------

def test_weighted_loss_is_positive(tmp_path):
    config = make_config(output_dir=str(tmp_path))
    model = make_model()
    trainer = Trainer(model, config)
    loader = make_loader()
    loss = trainer.train_epoch(loader)
    assert loss > 0.0


def test_weighted_loss_changes_with_different_weights(tmp_path):
    """Different loss weights should produce different total loss values."""
    torch.manual_seed(0)
    inputs = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, NUM_CLASSES, (4,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)

    num_exits = len(EXIT_INDICES) + 1

    config_equal = make_config(output_dir=str(tmp_path / "equal"))
    config_equal.exit_loss_weights = [1.0 / num_exits] * num_exits

    config_skewed = make_config(output_dir=str(tmp_path / "skewed"))
    config_skewed.exit_loss_weights = [0.8] + [0.2 / (num_exits - 1)] * (num_exits - 1)

    torch.manual_seed(1)
    model_a = make_model()
    trainer_a = Trainer(model_a, config_equal)
    loss_a = trainer_a.train_epoch(loader)

    torch.manual_seed(1)
    model_b = make_model()
    trainer_b = Trainer(model_b, config_skewed)
    loss_b = trainer_b.train_epoch(loader)

    assert loss_a != pytest.approx(loss_b)


def test_train_runs_for_configured_epochs(tmp_path):
    config = make_config(output_dir=str(tmp_path), lr_scheduler=None)
    config.num_epochs = 3
    model = make_model()
    trainer = Trainer(model, config)
    loader = make_loader()
    # Should complete without error.
    trainer.train(loader)


def test_train_num_epochs_override(tmp_path):
    config = make_config(output_dir=str(tmp_path))
    config.num_epochs = 10
    model = make_model()
    trainer = Trainer(model, config)
    loader = make_loader()
    # Override to 1 epoch; should not run 10.
    trainer.train(loader, num_epochs=1)
