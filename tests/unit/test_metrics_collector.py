import json
import os
import tempfile

import pytest

from src.metrics.collector import MetricsCollector
from src.metrics.types import AggregatedMetrics
from src.models.types import InferenceResult


def make_result(
    predicted_class: int,
    exit_layer: int,
    inference_time_ms: float = 1.0,
    flops_consumed: int = 100,
) -> InferenceResult:
    return InferenceResult(
        predicted_class=predicted_class,
        confidence=0.9,
        exit_layer=exit_layer,
        flops_consumed=flops_consumed,
        inference_time_ms=inference_time_ms,
    )


def test_aggregate_accuracy_and_exit_frequencies() -> None:
    collector = MetricsCollector()
    # 3 correct, 1 wrong; exits at layers 1, 1, 2, 2
    collector.record(make_result(0, exit_layer=1), ground_truth=0)
    collector.record(make_result(1, exit_layer=1), ground_truth=1)
    collector.record(make_result(2, exit_layer=2), ground_truth=2)
    collector.record(make_result(3, exit_layer=2), ground_truth=9)  # wrong

    metrics = collector.aggregate()

    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.exit_frequency[1] == pytest.approx(0.5)
    assert metrics.exit_frequency[2] == pytest.approx(0.5)


def test_exit_frequency_sums_to_one() -> None:
    collector = MetricsCollector()
    collector.record(make_result(0, exit_layer=1), ground_truth=0)
    collector.record(make_result(1, exit_layer=2), ground_truth=1)
    collector.record(make_result(2, exit_layer=3), ground_truth=2)
    collector.record(make_result(3, exit_layer=1), ground_truth=3)

    metrics = collector.aggregate()

    assert sum(metrics.exit_frequency.values()) == pytest.approx(1.0)


def test_aggregate_mean_flops_and_time() -> None:
    collector = MetricsCollector()
    collector.record(make_result(0, exit_layer=1, flops_consumed=200, inference_time_ms=2.0), ground_truth=0)
    collector.record(make_result(1, exit_layer=2, flops_consumed=400, inference_time_ms=4.0), ground_truth=1)

    metrics = collector.aggregate()

    assert metrics.mean_flops == pytest.approx(300.0)
    assert metrics.mean_inference_time_ms == pytest.approx(3.0)


def test_record_stores_all_required_fields() -> None:
    collector = MetricsCollector()
    result = InferenceResult(
        predicted_class=5,
        confidence=0.95,
        exit_layer=2,
        flops_consumed=512,
        inference_time_ms=3.7,
    )
    collector.record(result, ground_truth=5)

    stored_result, stored_gt = collector._records[0]
    assert stored_result.predicted_class == 5
    assert stored_result.exit_layer == 2
    assert stored_result.inference_time_ms == pytest.approx(3.7)
    assert stored_gt == 5


def test_json_round_trip() -> None:
    collector = MetricsCollector()
    collector.record(make_result(0, exit_layer=1, flops_consumed=100, inference_time_ms=1.5), ground_truth=0)
    collector.record(make_result(1, exit_layer=2, flops_consumed=200, inference_time_ms=2.5), ground_truth=0)  # wrong

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "metrics.json")
        collector.save_json(path)

        with open(path) as f:
            data = json.load(f)

    loaded = AggregatedMetrics(
        accuracy=data["accuracy"],
        mean_flops=data["mean_flops"],
        mean_inference_time_ms=data["mean_inference_time_ms"],
        exit_frequency={int(k): v for k, v in data["exit_frequency"].items()},
    )
    original = collector.aggregate()

    assert loaded.accuracy == pytest.approx(original.accuracy)
    assert loaded.mean_flops == pytest.approx(original.mean_flops)
    assert loaded.mean_inference_time_ms == pytest.approx(original.mean_inference_time_ms)
    assert loaded.exit_frequency == pytest.approx(original.exit_frequency)


def test_save_json_creates_parent_directories() -> None:
    collector = MetricsCollector()
    collector.record(make_result(0, exit_layer=1), ground_truth=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "nested", "dir", "metrics.json")
        collector.save_json(path)
        assert os.path.exists(path)


def test_aggregate_empty_collector() -> None:
    collector = MetricsCollector()
    metrics = collector.aggregate()

    assert metrics.accuracy == 0.0
    assert metrics.mean_flops == 0.0
    assert metrics.mean_inference_time_ms == 0.0
    assert metrics.exit_frequency == {}
