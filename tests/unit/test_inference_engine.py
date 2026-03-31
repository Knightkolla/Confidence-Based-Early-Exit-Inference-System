import torch
from torch.utils.data import DataLoader, TensorDataset

from src.engine.inference import InferenceEngine
from src.models.early_exit_model import EarlyExitModel
from src.models.mlp_backbone import MLPBackbone

NUM_CLASSES = 4
HIDDEN_DIM = 16
NUM_LAYERS = 3
EXIT_INDICES = [1, 2]


def make_model() -> EarlyExitModel:
    backbone = MLPBackbone(
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
    )
    return EarlyExitModel(
        backbone=backbone,
        exit_layer_indices=EXIT_INDICES,
        num_classes=NUM_CLASSES,
    )


def make_input() -> torch.Tensor:
    return torch.randn(1, 3, 32, 32)


# ---------------------------------------------------------------------------
# Threshold boundary
# ---------------------------------------------------------------------------

def test_threshold_exactly_equal_fires():
    """A head whose confidence equals the threshold exactly should trigger early exit."""
    model = make_model()
    engine = InferenceEngine(model)
    x = make_input()

    model.eval()
    with torch.no_grad():
        outputs = model(x)

    # Use the confidence of the first exit head as the threshold.
    first_confidence = float(outputs[0].confidence.item())
    result = engine.infer(x, threshold=first_confidence)

    assert result.exit_layer == outputs[0].exit_layer
    assert result.confidence == first_confidence


def test_threshold_just_below_fires_earlier():
    """A threshold just below the first head's confidence should exit at that head."""
    model = make_model()
    engine = InferenceEngine(model)
    x = make_input()

    model.eval()
    with torch.no_grad():
        outputs = model(x)

    first_confidence = float(outputs[0].confidence.item())
    # Threshold slightly below first head's confidence -> should exit at first head.
    result = engine.infer(x, threshold=first_confidence - 1e-6)
    assert result.exit_layer == outputs[0].exit_layer


def test_threshold_just_above_does_not_fire_at_first_head():
    """A threshold just above the first head's confidence should not exit at that head."""
    model = make_model()
    engine = InferenceEngine(model)
    x = make_input()

    model.eval()
    with torch.no_grad():
        outputs = model(x)

    first_confidence = float(outputs[0].confidence.item())
    # Threshold slightly above first head -> must not exit at first head.
    result = engine.infer(x, threshold=first_confidence + 1e-6)
    assert result.exit_layer != outputs[0].exit_layer


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------

def test_threshold_1_forces_fallback_to_final_layer():
    """threshold=1.0 means no softmax output can reach it; must fall back to final head."""
    model = make_model()
    engine = InferenceEngine(model)
    x = make_input()

    final_exit_layer = NUM_LAYERS + 1
    result = engine.infer(x, threshold=1.0)
    assert result.exit_layer == final_exit_layer


def test_threshold_0_exits_at_first_head():
    """threshold=0.0 means any confidence qualifies; should exit at the very first head."""
    model = make_model()
    engine = InferenceEngine(model)
    x = make_input()

    model.eval()
    with torch.no_grad():
        outputs = model(x)

    result = engine.infer(x, threshold=0.0)
    assert result.exit_layer == outputs[0].exit_layer


# ---------------------------------------------------------------------------
# FLOPs and timing
# ---------------------------------------------------------------------------

def test_flops_consumed_is_positive():
    model = make_model()
    engine = InferenceEngine(model)
    result = engine.infer(make_input(), threshold=0.5)
    assert result.flops_consumed > 0


def test_inference_time_ms_is_positive():
    model = make_model()
    engine = InferenceEngine(model)
    result = engine.infer(make_input(), threshold=0.5)
    assert result.inference_time_ms > 0.0


def test_flops_fallback_equals_last_layer_flops():
    """When falling back to the final head, flops should equal the last layer_flops entry."""
    model = make_model()
    engine = InferenceEngine(model)
    x = make_input()

    result = engine.infer(x, threshold=1.0)
    expected = model.layer_flops[-1].backbone_flops + model.layer_flops[-1].exit_head_flops
    assert result.flops_consumed == expected


# ---------------------------------------------------------------------------
# infer_batch
# ---------------------------------------------------------------------------

def test_infer_batch_returns_one_result_per_sample():
    model = make_model()
    engine = InferenceEngine(model)

    inputs = torch.randn(6, 3, 32, 32)
    targets = torch.zeros(6, dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    results = engine.infer_batch(loader, threshold=0.5)
    assert len(results) == 6


def test_infer_batch_threshold_1_all_use_final_layer():
    """With threshold=1.0 every sample must fall back to the final layer."""
    model = make_model()
    engine = InferenceEngine(model)

    inputs = torch.randn(4, 3, 32, 32)
    targets = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)

    final_exit_layer = NUM_LAYERS + 1
    results = engine.infer_batch(loader, threshold=1.0)
    assert all(r.exit_layer == final_exit_layer for r in results)


# ---------------------------------------------------------------------------
# 3-D input (C, H, W) without batch dimension
# ---------------------------------------------------------------------------

def test_infer_accepts_unbatched_input():
    """infer() should handle a (C, H, W) tensor by adding the batch dimension internally."""
    model = make_model()
    engine = InferenceEngine(model)
    x = torch.randn(3, 32, 32)  # no batch dim
    result = engine.infer(x, threshold=0.5)
    assert isinstance(result.predicted_class, int)
