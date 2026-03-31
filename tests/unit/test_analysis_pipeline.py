import csv
import os

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.analysis.pipeline import AnalysisPipeline
from src.metrics.types import TradeoffRow, TradeoffTable
from src.models.early_exit_model import EarlyExitModel
from src.models.mlp_backbone import MLPBackbone

NUM_CLASSES = 4
HIDDEN_DIM = 16
NUM_LAYERS = 3
EXIT_INDICES = [1, 2]
NUM_SAMPLES = 8
BATCH_SIZE = 4


def make_model() -> EarlyExitModel:
    backbone = MLPBackbone(
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
    )
    return EarlyExitModel(
        backbone=backbone,
        exit_layer_indices=EXIT_INDICES,
        num_classes=NUM_CLASSES,
    )


def make_loader() -> DataLoader:
    inputs = torch.randn(NUM_SAMPLES, 3, 32, 32)
    # Assign deterministic labels so accuracy is well-defined.
    targets = torch.arange(NUM_SAMPLES) % NUM_CLASSES
    return DataLoader(TensorDataset(inputs, targets), batch_size=BATCH_SIZE, shuffle=False)


def make_pipeline() -> AnalysisPipeline:
    return AnalysisPipeline(make_model(), make_loader())


# ---------------------------------------------------------------------------
# run_sweep
# ---------------------------------------------------------------------------

def test_run_sweep_one_row_per_threshold():
    pipeline = make_pipeline()
    thresholds = [0.5, 0.7, 0.9]
    table = pipeline.run_sweep(thresholds)
    # 1.0 is always added, so we expect len(thresholds) + 1 rows.
    assert len(table.rows) == len(thresholds) + 1


def test_run_sweep_baseline_always_present():
    pipeline = make_pipeline()
    table = pipeline.run_sweep([0.5, 0.7])
    thresholds_in_table = [row.threshold for row in table.rows]
    assert 1.0 in thresholds_in_table


def test_run_sweep_baseline_not_duplicated_when_included():
    pipeline = make_pipeline()
    table = pipeline.run_sweep([0.5, 1.0])
    count = sum(1 for row in table.rows if row.threshold == 1.0)
    assert count == 1


def test_run_sweep_rows_sorted_ascending():
    pipeline = make_pipeline()
    table = pipeline.run_sweep([0.9, 0.5, 0.7])
    thresholds = [row.threshold for row in table.rows]
    assert thresholds == sorted(thresholds)


def test_run_sweep_baseline_row_has_zero_reduction_and_drop():
    pipeline = make_pipeline()
    table = pipeline.run_sweep([0.5])
    baseline = next(r for r in table.rows if r.threshold == 1.0)
    assert baseline.flops_reduction == 0.0
    assert baseline.accuracy_drop == 0.0


def test_run_sweep_flops_reduction_computed_correctly():
    """flops_reduction = (baseline_flops - row_flops) / baseline_flops."""
    pipeline = make_pipeline()
    table = pipeline.run_sweep([0.0])
    baseline = next(r for r in table.rows if r.threshold == 1.0)
    early = next(r for r in table.rows if r.threshold == 0.0)

    if baseline.mean_flops > 0:
        expected = (baseline.mean_flops - early.mean_flops) / baseline.mean_flops
        assert abs(early.flops_reduction - expected) < 1e-9


def test_run_sweep_accuracy_drop_computed_correctly():
    """accuracy_drop = baseline_accuracy - row_accuracy."""
    pipeline = make_pipeline()
    table = pipeline.run_sweep([0.0])
    baseline = next(r for r in table.rows if r.threshold == 1.0)
    early = next(r for r in table.rows if r.threshold == 0.0)
    expected_drop = baseline.accuracy - early.accuracy
    assert abs(early.accuracy_drop - expected_drop) < 1e-9


def test_run_sweep_row_fields_populated():
    pipeline = make_pipeline()
    table = pipeline.run_sweep([0.5])
    for row in table.rows:
        assert row.accuracy >= 0.0
        assert row.mean_flops > 0
        assert row.mean_inference_time_ms > 0.0
        assert isinstance(row.exit_frequency, dict)


# ---------------------------------------------------------------------------
# save_csv
# ---------------------------------------------------------------------------

def test_save_csv_creates_file(tmp_path):
    pipeline = make_pipeline()
    table = pipeline.run_sweep([0.5])
    out = str(tmp_path / "results" / "table.csv")
    pipeline.save_csv(table, out)
    assert os.path.isfile(out)


def test_save_csv_round_trip(tmp_path):
    """Saving then loading the CSV should reproduce all scalar row values."""
    pipeline = make_pipeline()
    table = pipeline.run_sweep([0.5, 0.8])
    out = str(tmp_path / "table.csv")
    pipeline.save_csv(table, out)

    with open(out, newline="") as f:
        reader = csv.DictReader(f)
        loaded = list(reader)

    assert len(loaded) == len(table.rows)
    for orig, row_dict in zip(table.rows, loaded):
        assert abs(float(row_dict["threshold"]) - orig.threshold) < 1e-9
        assert abs(float(row_dict["accuracy"]) - orig.accuracy) < 1e-9
        assert abs(float(row_dict["mean_flops"]) - orig.mean_flops) < 1e-9
        assert abs(float(row_dict["flops_reduction"]) - orig.flops_reduction) < 1e-9
        assert abs(float(row_dict["accuracy_drop"]) - orig.accuracy_drop) < 1e-9


def test_save_csv_exit_freq_columns(tmp_path):
    """CSV must include exit_freq_<layer> columns for every exit layer seen."""
    pipeline = make_pipeline()
    table = pipeline.run_sweep([0.0])
    out = str(tmp_path / "table.csv")
    pipeline.save_csv(table, out)

    with open(out, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

    exit_freq_cols = [h for h in headers if h.startswith("exit_freq_")]
    assert len(exit_freq_cols) >= 1


def test_save_csv_creates_parent_dirs(tmp_path):
    pipeline = make_pipeline()
    table = pipeline.run_sweep([0.5])
    nested = str(tmp_path / "a" / "b" / "c" / "out.csv")
    pipeline.save_csv(table, nested)
    assert os.path.isfile(nested)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def test_plot_accuracy_vs_flops_creates_file(tmp_path):
    pipeline = make_pipeline()
    table = pipeline.run_sweep([0.5, 0.8])
    out = str(tmp_path / "plots" / "flops.png")
    pipeline.plot_accuracy_vs_flops(table, out)
    assert os.path.isfile(out)


def test_plot_accuracy_vs_time_creates_file(tmp_path):
    pipeline = make_pipeline()
    table = pipeline.run_sweep([0.5, 0.8])
    out = str(tmp_path / "plots" / "time.png")
    pipeline.plot_accuracy_vs_time(table, out)
    assert os.path.isfile(out)


def test_plots_create_parent_dirs(tmp_path):
    pipeline = make_pipeline()
    table = pipeline.run_sweep([0.5])
    flops_path = str(tmp_path / "deep" / "nested" / "flops.png")
    time_path = str(tmp_path / "deep" / "nested" / "time.png")
    pipeline.plot_accuracy_vs_flops(table, flops_path)
    pipeline.plot_accuracy_vs_time(table, time_path)
    assert os.path.isfile(flops_path)
    assert os.path.isfile(time_path)
