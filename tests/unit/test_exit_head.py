import math

import pytest
import torch

from src.config.errors import ConfigurationError
from src.models.early_exit_model import EarlyExitModel
from src.models.exit_head import ExitHead
from src.models.mlp_backbone import MLPBackbone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_exit_head(
    num_classes: int = 10,
    exit_layer: int = 1,
    confidence_method: str = "max_softmax",
    in_features: int = 64,
) -> ExitHead:
    return ExitHead(
        in_features=in_features,
        num_classes=num_classes,
        exit_layer=exit_layer,
        confidence_method=confidence_method,
    )


def make_mlp_model(
    num_layers: int = 4,
    num_classes: int = 10,
    exit_layer_indices: list[int] | None = None,
    confidence_method: str = "max_softmax",
) -> EarlyExitModel:
    if exit_layer_indices is None:
        exit_layer_indices = [1, 2]
    backbone = MLPBackbone(num_layers=num_layers, num_classes=num_classes, hidden_dim=64)
    return EarlyExitModel(
        backbone=backbone,
        exit_layer_indices=exit_layer_indices,
        num_classes=num_classes,
        confidence_method=confidence_method,
    )


# ---------------------------------------------------------------------------
# ExitHead: confidence range
# ---------------------------------------------------------------------------

def test_max_softmax_confidence_in_unit_interval():
    head = make_exit_head(num_classes=10, confidence_method="max_softmax")
    logits = torch.randn(8, 64)
    output = head(logits)
    assert output.confidence.shape == (8,)
    assert (output.confidence >= 0.0).all()
    assert (output.confidence <= 1.0).all()


def test_entropy_confidence_in_unit_interval():
    head = make_exit_head(num_classes=10, confidence_method="entropy")
    logits = torch.randn(8, 64)
    output = head(logits)
    assert output.confidence.shape == (8,)
    assert (output.confidence >= 0.0).all()
    assert (output.confidence <= 1.0).all()


# ---------------------------------------------------------------------------
# ExitHead: correctness of confidence values
# ---------------------------------------------------------------------------

def test_max_softmax_confidence_equals_max_softmax():
    head = make_exit_head(num_classes=5, confidence_method="max_softmax", in_features=5)
    # Use fixed weights so the linear layer is identity-like; bypass it by
    # setting weight=I and bias=0 so logits pass through unchanged.
    with torch.no_grad():
        head.linear.weight.copy_(torch.eye(5))
        head.linear.bias.zero_()

    raw = torch.tensor([[1.0, 2.0, 3.0, 0.5, -1.0]])
    output = head(raw)
    expected = torch.softmax(raw, dim=-1).max(dim=-1).values
    assert torch.allclose(output.confidence, expected, atol=1e-5)


def test_entropy_confidence_equals_formula():
    head = make_exit_head(num_classes=5, confidence_method="entropy", in_features=5)
    with torch.no_grad():
        head.linear.weight.copy_(torch.eye(5))
        head.linear.bias.zero_()

    raw = torch.tensor([[1.0, 2.0, 3.0, 0.5, -1.0]])
    output = head(raw)

    probs = torch.softmax(raw, dim=-1)
    entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1)
    expected = (1.0 - entropy / math.log(5)).clamp(0.0, 1.0)
    assert torch.allclose(output.confidence, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# ExitHead: predicted class
# ---------------------------------------------------------------------------

def test_predicted_class_equals_argmax():
    head = make_exit_head(num_classes=5, in_features=5)
    with torch.no_grad():
        head.linear.weight.copy_(torch.eye(5))
        head.linear.bias.zero_()

    raw = torch.tensor([[0.1, 0.9, 0.3, 0.2, 0.05]])
    output = head(raw)
    assert output.predicted_class.item() == raw.argmax(dim=-1).item()


def test_predicted_class_batch():
    head = make_exit_head(num_classes=4, in_features=4)
    with torch.no_grad():
        head.linear.weight.copy_(torch.eye(4))
        head.linear.bias.zero_()

    raw = torch.tensor([
        [0.1, 0.9, 0.3, 0.2],
        [0.5, 0.1, 0.8, 0.2],
    ])
    output = head(raw)
    expected = raw.argmax(dim=-1)
    assert torch.equal(output.predicted_class, expected)


# ---------------------------------------------------------------------------
# ExitHead: non-finite logit errors
# ---------------------------------------------------------------------------

def test_nan_logits_raise_runtime_error():
    head = make_exit_head(num_classes=4, exit_layer=3, in_features=4)
    with torch.no_grad():
        head.linear.weight.copy_(torch.eye(4))
        head.linear.bias.zero_()

    nan_input = torch.tensor([[float("nan"), 1.0, 2.0, 3.0]])
    with pytest.raises(RuntimeError):
        head(nan_input)


def test_inf_logits_raise_runtime_error():
    head = make_exit_head(num_classes=4, exit_layer=2, in_features=4)
    with torch.no_grad():
        head.linear.weight.copy_(torch.eye(4))
        head.linear.bias.zero_()

    inf_input = torch.tensor([[float("inf"), 1.0, 2.0, 3.0]])
    with pytest.raises(RuntimeError):
        head(inf_input)


def test_runtime_error_message_contains_exit_layer_index():
    exit_layer = 7
    head = make_exit_head(num_classes=4, exit_layer=exit_layer, in_features=4)
    with torch.no_grad():
        head.linear.weight.copy_(torch.eye(4))
        head.linear.bias.zero_()

    nan_input = torch.tensor([[float("nan"), 1.0, 2.0, 3.0]])
    with pytest.raises(RuntimeError, match=str(exit_layer)):
        head(nan_input)


# ---------------------------------------------------------------------------
# ExitHead: input dimensionality
# ---------------------------------------------------------------------------

def test_2d_input_batch_features():
    head = make_exit_head(num_classes=10, in_features=64)
    x = torch.randn(4, 64)
    output = head(x)
    assert output.logits.shape == (4, 10)
    assert output.confidence.shape == (4,)
    assert output.predicted_class.shape == (4,)


def test_3d_input_global_avg_pool():
    # (batch, seq_len, embed_dim) from Transformer -> pool over seq dim (dim=1)
    head = make_exit_head(num_classes=10, in_features=64)
    x = torch.randn(4, 16, 64)  # (batch, seq_len, embed_dim)
    output = head(x)
    assert output.logits.shape == (4, 10)
    assert output.confidence.shape == (4,)


def test_4d_input_global_avg_pool():
    # (batch, channels, h, w) -> global avg pool over dims (2, 3)
    head = make_exit_head(num_classes=10, in_features=64)
    x = torch.randn(4, 64, 8, 8)
    output = head(x)
    assert output.logits.shape == (4, 10)
    assert output.confidence.shape == (4,)


# ---------------------------------------------------------------------------
# EarlyExitModel: ConfigurationError on invalid indices
# ---------------------------------------------------------------------------

def test_configuration_error_exit_index_exceeds_depth():
    backbone = MLPBackbone(num_layers=3, num_classes=10, hidden_dim=64)
    with pytest.raises(ConfigurationError):
        EarlyExitModel(
            backbone=backbone,
            exit_layer_indices=[4],
            num_classes=10,
        )


def test_configuration_error_exit_index_zero():
    backbone = MLPBackbone(num_layers=3, num_classes=10, hidden_dim=64)
    with pytest.raises(ConfigurationError):
        EarlyExitModel(
            backbone=backbone,
            exit_layer_indices=[0],
            num_classes=10,
        )


def test_configuration_error_exit_index_negative():
    backbone = MLPBackbone(num_layers=3, num_classes=10, hidden_dim=64)
    with pytest.raises(ConfigurationError):
        EarlyExitModel(
            backbone=backbone,
            exit_layer_indices=[-1],
            num_classes=10,
        )


def test_valid_exit_indices_do_not_raise():
    backbone = MLPBackbone(num_layers=4, num_classes=10, hidden_dim=64)
    model = EarlyExitModel(
        backbone=backbone,
        exit_layer_indices=[1, 2, 3, 4],
        num_classes=10,
    )
    assert model is not None


# ---------------------------------------------------------------------------
# EarlyExitModel: forward() output structure
# ---------------------------------------------------------------------------

def test_forward_returns_list_of_exit_outputs_correct_length():
    model = make_mlp_model(num_layers=4, exit_layer_indices=[1, 3])
    x = torch.randn(2, 3, 32, 32)
    outputs = model(x)
    # 2 exit heads + 1 final head
    assert len(outputs) == 3


def test_forward_single_exit_head_plus_final():
    model = make_mlp_model(num_layers=4, exit_layer_indices=[2])
    x = torch.randn(2, 3, 32, 32)
    outputs = model(x)
    assert len(outputs) == 2


def test_forward_outputs_in_ascending_layer_order():
    model = make_mlp_model(num_layers=4, exit_layer_indices=[1, 3])
    x = torch.randn(2, 3, 32, 32)
    outputs = model(x)
    layers = [o.exit_layer for o in outputs]
    assert layers == sorted(layers)


def test_forward_exit_layers_match_configured_indices():
    exit_indices = [1, 3]
    model = make_mlp_model(num_layers=4, exit_layer_indices=exit_indices)
    x = torch.randn(2, 3, 32, 32)
    outputs = model(x)
    # First len(exit_indices) outputs correspond to configured exit heads
    for i, idx in enumerate(sorted(exit_indices)):
        assert outputs[i].exit_layer == idx


def test_exit_heads_attached_at_exactly_configured_indices():
    exit_indices = [2, 4]
    backbone = MLPBackbone(num_layers=4, num_classes=10, hidden_dim=64)
    model = EarlyExitModel(
        backbone=backbone,
        exit_layer_indices=exit_indices,
        num_classes=10,
    )
    assert set(model.exit_heads.keys()) == {str(i) for i in exit_indices}


def test_final_head_exit_layer_equals_backbone_depth_plus_one():
    num_layers = 4
    model = make_mlp_model(num_layers=num_layers, exit_layer_indices=[1, 2])
    x = torch.randn(2, 3, 32, 32)
    outputs = model(x)
    final_output = outputs[-1]
    assert final_output.exit_layer == num_layers + 1


def test_forward_confidence_in_unit_interval():
    model = make_mlp_model(num_layers=4, exit_layer_indices=[1, 2])
    x = torch.randn(4, 3, 32, 32)
    outputs = model(x)
    for output in outputs:
        assert (output.confidence >= 0.0).all()
        assert (output.confidence <= 1.0).all()


def test_forward_entropy_mode_confidence_in_unit_interval():
    model = make_mlp_model(
        num_layers=4, exit_layer_indices=[1, 2], confidence_method="entropy"
    )
    x = torch.randn(4, 3, 32, 32)
    outputs = model(x)
    for output in outputs[:-1]:  # exit heads only; final head always uses max_softmax
        assert (output.confidence >= 0.0).all()
        assert (output.confidence <= 1.0).all()
