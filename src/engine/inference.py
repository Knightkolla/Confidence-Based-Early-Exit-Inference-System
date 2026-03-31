import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.models.early_exit_model import EarlyExitModel
from src.models.types import InferenceResult


class InferenceEngine:
    def __init__(self, model: EarlyExitModel) -> None:
        self.model = model

    def infer(self, x: Tensor, threshold: float) -> InferenceResult:
        """Run single-sample inference, exiting at the first head whose confidence >= threshold."""
        if x.dim() == 3:
            x = x.unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            start = time.perf_counter()
            exit_outputs = self.model(x)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

        layer_flops = self.model.layer_flops

        for output in exit_outputs:
            confidence_val = output.confidence.item()
            if confidence_val >= threshold:
                flops = self._flops_for_layer(output.exit_layer, layer_flops)
                return InferenceResult(
                    predicted_class=int(output.predicted_class.item()),
                    confidence=confidence_val,
                    exit_layer=output.exit_layer,
                    flops_consumed=flops,
                    inference_time_ms=elapsed_ms,
                )

        # Fallback: use the final head (last element, always present)
        final = exit_outputs[-1]
        flops = self._flops_for_layer(final.exit_layer, layer_flops)
        return InferenceResult(
            predicted_class=int(final.predicted_class.item()),
            confidence=float(final.confidence.item()),
            exit_layer=final.exit_layer,
            flops_consumed=flops,
            inference_time_ms=elapsed_ms,
        )

    def infer_batch(self, data_loader: DataLoader, threshold: float) -> list[InferenceResult]:
        """Run inference on every sample in data_loader, one sample at a time."""
        results: list[InferenceResult] = []
        for batch in data_loader:
            # Support (inputs, targets) tuples or plain input tensors
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            for i in range(inputs.size(0)):
                results.append(self.infer(inputs[i], threshold))
        return results

    @staticmethod
    def _flops_for_layer(exit_layer: int, layer_flops: list) -> int:
        """Return total FLOPs consumed up to and including the given exit layer.

        exit_layer is 1-based. The final head uses index len(backbone.layers) + 1,
        which maps to the last entry in layer_flops.
        """
        # layer_flops is indexed 0-based; exit_layer is 1-based.
        # The final head's exit_layer exceeds len(layer_flops), so clamp to last entry.
        idx = min(exit_layer - 1, len(layer_flops) - 1)
        entry = layer_flops[idx]
        return entry.backbone_flops + entry.exit_head_flops
