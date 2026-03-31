"""
Integration test: full pipeline from training through analysis output.

Exercises Requirements 6.1-6.5 and 8.2 without downloading real CIFAR-10 data.
"""

import csv
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.analysis.pipeline import AnalysisPipeline
from src.config.types import AugmentationConfig, ExperimentConfig
from src.engine.trainer import Trainer
from src.models.early_exit_model import EarlyExitModel
from src.models.mlp_backbone import MLPBackbone

NUM_CLASSES = 10
NUM_SAMPLES = 16
BATCH_SIZE = 4
HIDDEN_DIM = 32
NUM_LAYERS = 4
EXIT_INDICES = [1, 2]
# one weight per exit head + final head
_NUM_HEADS = len(EXIT_INDICES) + 1
LOSS_WEIGHTS = [1.0 / _NUM_HEADS] * _NUM_HEADS
THRESHOLDS = [0.5, 0.7, 0.9]


def _make_synthetic_loader() -> DataLoader:
    torch.manual_seed(0)
    inputs = torch.randn(NUM_SAMPLES, 3, 32, 32)
    targets = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    return DataLoader(
        TensorDataset(inputs, targets),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )


def _make_model() -> EarlyExitModel:
    backbone = MLPBackbone(
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
    )
    return EarlyExitModel(
        backbone=backbone,
        exit_layer_indices=EXIT_INDICES,
        num_classes=NUM_CLASSES,
        confidence_method="max_softmax",
    )


def _make_config(output_dir: str) -> ExperimentConfig:
    return ExperimentConfig(
        backbone="mlp",
        exit_layer_indices=EXIT_INDICES,
        confidence_method="max_softmax",
        exit_loss_weights=LOSS_WEIGHTS,
        optimizer="adam",
        learning_rate=1e-3,
        lr_scheduler=None,
        num_epochs=1,
        batch_size=BATCH_SIZE,
        threshold_sweep=THRESHOLDS,
        dataset="cifar10",
        dataset_path="/tmp/unused",
        augmentation=AugmentationConfig(),
        random_seed=42,
        output_dir=output_dir,
    )


@pytest.fixture()
def pipeline_outputs(tmp_path: Path):
    """Run the full pipeline and return (output_dir, table)."""
    output_dir = str(tmp_path)
    config = _make_config(output_dir)
    model = _make_model()

    loader = _make_synthetic_loader()

    trainer = Trainer(model, config)
    trainer.train(loader, num_epochs=1)

    pipeline = AnalysisPipeline(model, loader)
    table = pipeline.run_sweep(THRESHOLDS)

    pipeline.save_csv(table, str(tmp_path / "tradeoff_table.csv"))
    pipeline.plot_accuracy_vs_flops(table, str(tmp_path / "accuracy_vs_flops.png"))
    pipeline.plot_accuracy_vs_time(table, str(tmp_path / "accuracy_vs_time.png"))

    return tmp_path, table


# ---------------------------------------------------------------------------
# Output file existence (Req 8.2, 6.4, 6.5)
# ---------------------------------------------------------------------------

def test_config_yaml_exists(pipeline_outputs):
    output_dir, _ = pipeline_outputs
    assert (output_dir / "config.yaml").exists()


def test_tradeoff_csv_exists(pipeline_outputs):
    output_dir, _ = pipeline_outputs
    assert (output_dir / "tradeoff_table.csv").exists()


def test_accuracy_vs_flops_plot_exists(pipeline_outputs):
    output_dir, _ = pipeline_outputs
    assert (output_dir / "accuracy_vs_flops.png").exists()


def test_accuracy_vs_time_plot_exists(pipeline_outputs):
    output_dir, _ = pipeline_outputs
    assert (output_dir / "accuracy_vs_time.png").exists()


# ---------------------------------------------------------------------------
# Trade-off table structure (Req 6.1, 6.2, 6.3)
# ---------------------------------------------------------------------------

def test_table_contains_baseline_row(pipeline_outputs):
    _, table = pipeline_outputs
    thresholds = [row.threshold for row in table.rows]
    assert 1.0 in thresholds


def test_table_row_count(pipeline_outputs):
    """One row per unique threshold; 1.0 is always appended."""
    _, table = pipeline_outputs
    expected = len(set(THRESHOLDS) | {1.0})
    assert len(table.rows) == expected


def test_table_rows_sorted_ascending(pipeline_outputs):
    _, table = pipeline_outputs
    thresholds = [row.threshold for row in table.rows]
    assert thresholds == sorted(thresholds)


def test_baseline_row_zero_flops_reduction(pipeline_outputs):
    _, table = pipeline_outputs
    baseline = next(r for r in table.rows if r.threshold == 1.0)
    assert baseline.flops_reduction == pytest.approx(0.0)


def test_baseline_row_zero_accuracy_drop(pipeline_outputs):
    _, table = pipeline_outputs
    baseline = next(r for r in table.rows if r.threshold == 1.0)
    assert baseline.accuracy_drop == pytest.approx(0.0)


def test_all_rows_have_positive_flops(pipeline_outputs):
    _, table = pipeline_outputs
    for row in table.rows:
        assert row.mean_flops > 0


def test_all_rows_have_valid_accuracy(pipeline_outputs):
    _, table = pipeline_outputs
    for row in table.rows:
        assert 0.0 <= row.accuracy <= 1.0


# ---------------------------------------------------------------------------
# CSV content (Req 6.4)
# ---------------------------------------------------------------------------

def test_csv_has_required_columns(pipeline_outputs):
    output_dir, _ = pipeline_outputs
    with open(output_dir / "tradeoff_table.csv", newline="") as f:
        headers = csv.DictReader(f).fieldnames or []
    required = {
        "threshold",
        "accuracy",
        "mean_flops",
        "mean_inference_time_ms",
        "flops_reduction",
        "accuracy_drop",
    }
    assert required.issubset(set(headers))


def test_csv_row_count_matches_table(pipeline_outputs):
    output_dir, table = pipeline_outputs
    with open(output_dir / "tradeoff_table.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(table.rows)


def test_csv_baseline_row_present(pipeline_outputs):
    output_dir, _ = pipeline_outputs
    with open(output_dir / "tradeoff_table.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    thresholds = [float(r["threshold"]) for r in rows]
    assert 1.0 in thresholds
