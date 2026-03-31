from dataclasses import dataclass

from torch import Tensor


@dataclass
class ExitOutput:
    exit_layer: int       # 1-based index
    logits: Tensor        # shape: (batch, num_classes)
    confidence: Tensor    # shape: (batch,) in [0, 1]
    predicted_class: Tensor  # shape: (batch,)


@dataclass
class InferenceResult:
    predicted_class: int
    confidence: float
    exit_layer: int       # index of exit used; final layer index if no early exit
    flops_consumed: int
    inference_time_ms: float
