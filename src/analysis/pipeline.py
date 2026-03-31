import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.engine.inference import InferenceEngine
from src.metrics.collector import MetricsCollector
from src.metrics.types import TradeoffRow, TradeoffTable
from src.models.early_exit_model import EarlyExitModel


class AnalysisPipeline:
    def __init__(self, model: EarlyExitModel, data_loader: DataLoader) -> None:
        self._engine = InferenceEngine(model)
        self._data_loader = data_loader

    def run_sweep(self, thresholds: list[float]) -> TradeoffTable:
        all_thresholds = sorted(set(thresholds) | {1.0})

        # Collect ground-truth labels once; infer_batch re-iterates the loader per threshold.
        # We need ground truths aligned with the order infer_batch processes samples.
        ground_truths: list[int] = []
        for batch in self._data_loader:
            targets = batch[1] if isinstance(batch, (list, tuple)) else batch
            for i in range(targets.size(0)):
                ground_truths.append(int(targets[i].item()))

        rows: list[TradeoffRow] = []
        baseline_accuracy: float | None = None
        baseline_flops: float | None = None

        for threshold in all_thresholds:
            collector = MetricsCollector()
            results = self._engine.infer_batch(self._data_loader, threshold)
            for result, gt in zip(results, ground_truths):
                collector.record(result, gt)
            metrics = collector.aggregate()

            if threshold == 1.0:
                baseline_accuracy = metrics.accuracy
                baseline_flops = metrics.mean_flops

            rows.append(TradeoffRow(
                threshold=threshold,
                accuracy=metrics.accuracy,
                mean_flops=metrics.mean_flops,
                mean_inference_time_ms=metrics.mean_inference_time_ms,
                exit_frequency=metrics.exit_frequency,
                # Filled in after baseline is known.
                flops_reduction=0.0,
                accuracy_drop=0.0,
            ))

        # Compute relative metrics against the baseline row.
        for row in rows:
            if row.threshold == 1.0:
                row.flops_reduction = 0.0
                row.accuracy_drop = 0.0
            else:
                row.flops_reduction = (
                    (baseline_flops - row.mean_flops) / baseline_flops
                    if baseline_flops and baseline_flops > 0
                    else 0.0
                )
                row.accuracy_drop = (baseline_accuracy or 0.0) - row.accuracy

        table = TradeoffTable(rows=sorted(rows, key=lambda r: r.threshold))
        return table

    def save_csv(self, table: TradeoffTable, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Collect all exit layer keys across rows for consistent column ordering.
        all_exit_layers: list[int] = sorted(
            {layer for row in table.rows for layer in row.exit_frequency}
        )

        fieldnames = [
            "threshold",
            "accuracy",
            "mean_flops",
            "mean_inference_time_ms",
            "flops_reduction",
            "accuracy_drop",
        ] + [f"exit_freq_{layer}" for layer in all_exit_layers]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in table.rows:
                record: dict = {
                    "threshold": row.threshold,
                    "accuracy": row.accuracy,
                    "mean_flops": row.mean_flops,
                    "mean_inference_time_ms": row.mean_inference_time_ms,
                    "flops_reduction": row.flops_reduction,
                    "accuracy_drop": row.accuracy_drop,
                }
                for layer in all_exit_layers:
                    record[f"exit_freq_{layer}"] = row.exit_frequency.get(layer, 0.0)
                writer.writerow(record)

    def plot_accuracy_vs_flops(self, table: TradeoffTable, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig, ax = plt.subplots()
        flops = [row.mean_flops for row in table.rows]
        accuracies = [row.accuracy for row in table.rows]
        ax.plot(flops, accuracies, marker="o")
        for row in table.rows:
            ax.annotate(f"{row.threshold:.2f}", (row.mean_flops, row.accuracy),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel("Mean FLOPs per Sample")
        ax.set_ylabel("Accuracy")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def plot_accuracy_vs_time(self, table: TradeoffTable, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig, ax = plt.subplots()
        times = [row.mean_inference_time_ms for row in table.rows]
        accuracies = [row.accuracy for row in table.rows]
        ax.plot(times, accuracies, marker="o")
        for row in table.rows:
            ax.annotate(f"{row.threshold:.2f}", (row.mean_inference_time_ms, row.accuracy),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel("Mean Inference Time (ms)")
        ax.set_ylabel("Accuracy")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
