import pytest
import torch

from src.models.transformer_backbone import TransformerBackbone
from src.models.mlp_backbone import MLPBackbone


# ---- TransformerBackbone ----

def test_transformer_forward_output_shape():
    model = TransformerBackbone(num_layers=2, num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 10)


def test_transformer_forward_features_length():
    model = TransformerBackbone(num_layers=3, num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    feats = model.forward_features(x)
    assert len(feats) == 3


def test_transformer_forward_features_shape():
    model = TransformerBackbone(num_layers=2, num_classes=10, patch_size=4, embed_dim=64)
    x = torch.randn(2, 3, 32, 32)
    feats = model.forward_features(x)
    # seq_len = (32/4)^2 = 64
    assert feats[0].shape == (2, 64, 64)


def test_transformer_layers_is_module_list():
    model = TransformerBackbone(num_layers=4, num_classes=10)
    assert isinstance(model.layers, torch.nn.ModuleList)
    assert len(model.layers) == 4


def test_transformer_layer_flops_count():
    model = TransformerBackbone(num_layers=5, num_classes=10)
    assert len(model.layer_flops) == 5


def test_transformer_layer_flops_indices():
    model = TransformerBackbone(num_layers=3, num_classes=10)
    for i, lf in enumerate(model.layer_flops):
        assert lf.layer_index == i + 1


def test_transformer_layer_flops_cumulative():
    model = TransformerBackbone(num_layers=4, num_classes=10)
    flops = [lf.backbone_flops for lf in model.layer_flops]
    assert flops == sorted(flops)
    assert len(set(flops)) == len(flops)


def test_transformer_layer_flops_positive():
    model = TransformerBackbone(num_layers=2, num_classes=10)
    for lf in model.layer_flops:
        assert lf.backbone_flops > 0
        assert lf.exit_head_flops > 0


def test_transformer_cifar100():
    model = TransformerBackbone(num_layers=2, num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    assert model(x).shape == (2, 100)


def test_transformer_invalid_patch_size():
    with pytest.raises(ValueError):
        TransformerBackbone(num_layers=2, num_classes=10, image_size=32, patch_size=5)


# ---- MLPBackbone ----

def test_mlp_forward_output_shape():
    model = MLPBackbone(num_layers=3, num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 10)


def test_mlp_forward_features_length():
    model = MLPBackbone(num_layers=4, num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    feats = model.forward_features(x)
    assert len(feats) == 4


def test_mlp_forward_features_shape():
    model = MLPBackbone(num_layers=2, num_classes=10, hidden_dim=256)
    x = torch.randn(2, 3, 32, 32)
    feats = model.forward_features(x)
    assert feats[0].shape == (2, 256)


def test_mlp_layers_is_module_list():
    model = MLPBackbone(num_layers=3, num_classes=10)
    assert isinstance(model.layers, torch.nn.ModuleList)
    assert len(model.layers) == 3


def test_mlp_layer_flops_count():
    model = MLPBackbone(num_layers=5, num_classes=10)
    assert len(model.layer_flops) == 5


def test_mlp_layer_flops_indices():
    model = MLPBackbone(num_layers=3, num_classes=10)
    for i, lf in enumerate(model.layer_flops):
        assert lf.layer_index == i + 1


def test_mlp_layer_flops_cumulative():
    model = MLPBackbone(num_layers=4, num_classes=10)
    flops = [lf.backbone_flops for lf in model.layer_flops]
    assert flops == sorted(flops)
    assert len(set(flops)) == len(flops)


def test_mlp_layer_flops_positive():
    model = MLPBackbone(num_layers=2, num_classes=10)
    for lf in model.layer_flops:
        assert lf.backbone_flops > 0
        assert lf.exit_head_flops > 0


def test_mlp_cifar100():
    model = MLPBackbone(num_layers=2, num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    assert model(x).shape == (2, 100)


def test_mlp_first_layer_flops_analytical():
    # First layer: 2 * (3*32*32) * hidden_dim
    model = MLPBackbone(num_layers=2, num_classes=10, image_size=32, hidden_dim=512)
    expected = 2 * (3 * 32 * 32) * 512
    assert model.layer_flops[0].backbone_flops == expected


def test_mlp_second_layer_flops_analytical():
    # Second layer adds: 2 * hidden_dim * hidden_dim
    model = MLPBackbone(num_layers=2, num_classes=10, image_size=32, hidden_dim=512)
    expected = 2 * (3 * 32 * 32) * 512 + 2 * 512 * 512
    assert model.layer_flops[1].backbone_flops == expected
