import json
import os
from dataclasses import asdict

from src.metrics.types import AggregatedMetrics
from src.models.types import InferenceResult


class MetricsCollector:
    def __init__(self) -> None:
        self._records: list[tuple[InferenceResult, int]] = []

    def record(self, result: InferenceResult, ground_truth: int) -> None:
        self._records.append((result, ground_truth))

    def aggregate(self) -> AggregatedMetrics:
        if not self._records:
            return AggregatedMetrics(
                accuracy=0.0,
                mean_flops=0.0,
                mean_inference_time_ms=0.0,
                exit_frequency={},
            )

        n = len(self._records)
        correct = sum(
            1 for result, gt in self._records if result.predicted_class == gt
        )
        total_flops = sum(result.flops_consumed for result, _ in self._records)
        total_time = sum(result.inference_time_ms for result, _ in self._records)

        exit_counts: dict[int, int] = {}
        for result, _ in self._records:
            exit_counts[result.exit_layer] = exit_counts.get(result.exit_layer, 0) + 1

        exit_frequency = {layer: count / n for layer, count in exit_counts.items()}

        return AggregatedMetrics(
            accuracy=correct / n,
            mean_flops=total_flops / n,
            mean_inference_time_ms=total_time / n,
            exit_frequency=exit_frequency,
        )

    def save_json(self, path: str) -> None:
        metrics = self.aggregate()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(metrics), f, indent=2)
