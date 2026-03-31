import torch
import torch.nn as nn
from torch import Tensor

from src.config.errors import ConfigurationError
from src.metrics.types import LayerFLOPs
from src.models.exit_head import ExitHead
from src.models.types import ExitOutput


class EarlyExitModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        exit_layer_indices: list[int],
        num_classes: int,
        confidence_method: str = "max_softmax",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.exit_layer_indices = sorted(exit_layer_indices)
        self.layer_flops: list[LayerFLOPs] = backbone.layer_flops

        depth = len(backbone.layers)
        for idx in exit_layer_indices:
            if idx < 1 or idx > depth:
                raise ConfigurationError(
                    f"Exit layer index {idx} is out of range: backbone has {depth} layers "
                    f"(valid indices are 1 to {depth})."
                )

        # Derive feature dimension from the backbone's final linear head.
        # Both TransformerBackbone and MLPBackbone expose `head: nn.Linear`.
        if not hasattr(backbone, "head") or not isinstance(backbone.head, nn.Linear):
            raise ConfigurationError(
                "Backbone must expose 'head' as an nn.Linear to determine ExitHead input dimension."
            )
        in_features: int = backbone.head.in_features

        # Store exit heads in a ModuleDict so PyTorch tracks their parameters.
        # Keys are stringified layer indices because ModuleDict requires string keys.
        self.exit_heads = nn.ModuleDict({
            str(idx): ExitHead(
                in_features=in_features,
                num_classes=num_classes,
                exit_layer=idx,
                confidence_method=confidence_method,
            )
            for idx in self.exit_layer_indices
        })

    def forward(self, x: Tensor) -> list[ExitOutput]:
        features = self.backbone.forward_features(x)

        outputs: list[ExitOutput] = []
        for idx in self.exit_layer_indices:
            # features is 0-indexed; exit indices are 1-based.
            head = self.exit_heads[str(idx)]
            outputs.append(head(features[idx - 1]))

        # Final backbone head on the last feature map.
        final_features = features[-1]
        # Reduce sequence dimension for Transformer (3-D) features before the linear head.
        if final_features.dim() == 3:
            final_features = final_features.mean(dim=1)

        final_logits = self.backbone.head(final_features)
        final_probs = torch.softmax(final_logits, dim=-1)
        final_confidence = final_probs.max(dim=-1).values
        final_predicted = final_logits.argmax(dim=-1)

        final_exit_layer = len(self.backbone.layers) + 1
        outputs.append(ExitOutput(
            exit_layer=final_exit_layer,
            logits=final_logits,
            confidence=final_confidence,
            predicted_class=final_predicted,
        ))

        return outputs
