import torch.nn as nn
from torch import Tensor

from src.metrics.types import LayerFLOPs


class MLPBackbone(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_classes: int,
        image_size: int = 32,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        input_dim = 3 * image_size * image_size

        # First layer projects from input_dim to hidden_dim; subsequent layers are hidden_dim -> hidden_dim.
        def make_block(in_dim: int, out_dim: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())

        layer_list: list[nn.Module] = [make_block(input_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            layer_list.append(make_block(hidden_dim, hidden_dim))

        self.layers = nn.ModuleList(layer_list)
        self.head = nn.Linear(hidden_dim, num_classes)

        # Analytical FLOPs: each linear layer = 2 * in_dim * out_dim (multiply-add).
        exit_head_flops = 2 * hidden_dim * num_classes

        self.layer_flops: list[LayerFLOPs] = []
        cumulative = 0
        for i, in_dim in enumerate([input_dim] + [hidden_dim] * (num_layers - 1)):
            cumulative += 2 * in_dim * hidden_dim
            self.layer_flops.append(
                LayerFLOPs(
                    layer_index=i + 1,
                    backbone_flops=cumulative,
                    exit_head_flops=exit_head_flops,
                )
            )

    def forward_features(self, x: Tensor) -> list[Tensor]:
        """Return feature tensors after each MLP block."""
        x = x.flatten(1)
        features: list[Tensor] = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        return self.head(features[-1])
