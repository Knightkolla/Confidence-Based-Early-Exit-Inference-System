import math

import torch
import torch.nn as nn
from torch import Tensor

from src.models.types import ExitOutput


class ExitHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        exit_layer: int,
        confidence_method: str = "max_softmax",
    ) -> None:
        super().__init__()
        if confidence_method not in ("max_softmax", "entropy"):
            raise ValueError(
                f"confidence_method must be 'max_softmax' or 'entropy', got '{confidence_method}'"
            )
        self.exit_layer = exit_layer
        self.num_classes = num_classes
        self.confidence_method = confidence_method
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, features: Tensor) -> ExitOutput:
        if features.dim() == 3:
            # (batch, seq_len, embed_dim) from Transformer — pool over sequence
            x = features.mean(dim=1)
        elif features.dim() == 4:
            # (batch, channels, h, w) from CNN — pool over spatial dims
            x = features.mean(dim=(2, 3))
        else:
            x = features

        logits = self.linear(x)

        if not torch.isfinite(logits).all():
            raise RuntimeError(
                f"ExitHead at exit layer {self.exit_layer} produced non-finite logits"
            )

        probs = torch.softmax(logits, dim=-1)

        if self.confidence_method == "max_softmax":
            confidence = probs.max(dim=-1).values
        else:
            # Normalized entropy: 1 - H(p) / log(C), clamped to [0, 1] for numerical safety
            entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1)
            confidence = 1.0 - entropy / math.log(self.num_classes)
            confidence = confidence.clamp(0.0, 1.0)

        predicted_class = logits.argmax(dim=-1)

        return ExitOutput(
            exit_layer=self.exit_layer,
            logits=logits,
            confidence=confidence,
            predicted_class=predicted_class,
        )
