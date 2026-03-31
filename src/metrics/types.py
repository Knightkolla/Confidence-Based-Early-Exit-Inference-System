from dataclasses import dataclass, field


@dataclass
class AggregatedMetrics:
    accuracy: float
    mean_flops: float
    mean_inference_time_ms: float
    exit_frequency: dict[int, float]  # exit_layer -> fraction of samples


@dataclass
class LayerFLOPs:
    layer_index: int
    backbone_flops: int
    exit_head_flops: int


@dataclass
class TradeoffRow:
    threshold: float
    accuracy: float
    mean_flops: float
    mean_inference_time_ms: float
    exit_frequency: dict[int, float]  # exit_layer -> fraction of samples
    flops_reduction: float            # (baseline_flops - mean_flops) / baseline_flops
    accuracy_drop: float              # baseline_accuracy - accuracy


@dataclass
class TradeoffTable:
    rows: list[TradeoffRow] = field(default_factory=list)
