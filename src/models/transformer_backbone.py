import math

import torch
import torch.nn as nn
from torch import Tensor

from src.metrics.types import LayerFLOPs


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_classes: int,
        image_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 128,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size {image_size} must be divisible by patch_size {patch_size}"
            )

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2

        # Patch embedding: project each patch to embed_dim
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        mlp_dim = int(embed_dim * mlp_ratio)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Analytical FLOPs per transformer encoder layer.
        # Attention: 4 * seq_len * embed_dim^2 (Q/K/V projections + output projection)
        # FFN: 2 * seq_len * embed_dim * mlp_dim (two linear layers)
        seq_len = num_patches
        attn_flops = 4 * seq_len * embed_dim * embed_dim
        ffn_flops = 2 * seq_len * embed_dim * mlp_dim
        layer_flops_each = attn_flops + ffn_flops

        # exit_head_flops: global-average-pool is free; one linear layer over embed_dim
        exit_head_flops = 2 * embed_dim * num_classes

        self.layer_flops: list[LayerFLOPs] = [
            LayerFLOPs(
                layer_index=i + 1,
                backbone_flops=layer_flops_each * (i + 1),
                exit_head_flops=exit_head_flops,
            )
            for i in range(num_layers)
        ]

    def _embed(self, x: Tensor) -> Tensor:
        # x: (B, 3, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)          # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        return x

    def forward_features(self, x: Tensor) -> list[Tensor]:
        """Return feature tensors after each transformer encoder layer."""
        x = self._embed(x)
        features: list[Tensor] = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        x = self.norm(features[-1])
        x = x.mean(dim=1)  # global average pool over sequence
        return self.head(x)
