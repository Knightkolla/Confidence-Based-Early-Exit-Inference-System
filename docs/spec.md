# Confidence-Based-Early-Exit-Inference-System — Project Specification

This document consolidates the requirements, design, and implementation plan for the confidence-based early exit system. It serves as the authoritative reference for the project's intended behavior, architecture, and correctness guarantees.

---

## 1. Research Goal

The system investigates: **how effectively can confidence-based early stopping reduce computation while preserving accuracy in deep models?**

A neural network is augmented with lightweight classifier heads ("exit heads") at intermediate layers. During inference, each head computes a confidence score. If confidence exceeds a configurable threshold, the model returns that prediction immediately — skipping all remaining layers. A threshold sweep then characterizes the full accuracy vs. FLOPs trade-off curve.

Two backbone tracks are supported:
- **Option A**: CNN on CIFAR-10 (fast baseline)
- **Option B**: Transformer/MLP on CIFAR-10/100 (primary, implemented)

---

## 2. Glossary

| Term | Definition |
|---|---|
| EarlyExitModel | Neural network augmented with intermediate exit points |
| ExitHead | Lightweight classifier at an intermediate layer; produces confidence + prediction |
| ConfidenceScore | Scalar in [0, 1] representing model certainty; computed via max-softmax or normalized entropy |
| ExitThreshold | Configurable scalar in (0, 1]; if confidence ≥ threshold, exit fires |
| InferenceEngine | Runtime component applying threshold decisions in layer order |
| MetricsCollector | Records accuracy, FLOPs, inference time, exit frequency per sample |
| AnalysisPipeline | Aggregates metrics across threshold sweep; produces trade-off reports |
| FLOPs | Floating point operations; proxy for computational cost |
| ExitLayer | 1-based index of the layer at which an early exit decision is made |
| Backbone | Base neural network (CNN, Transformer, or MLP) without exit heads |

---

## 3. Requirements

### Req 1 — Early Exit Model Construction
- Attach ≥2 ExitHeads at distinct intermediate layers
- Exit layer indices are configurable (1-based)
- Raise `ConfigurationError` at init if any index exceeds backbone depth

### Req 2 — Confidence Score Computation
- Default: `max(softmax(logits))` — max-softmax probability
- Alternative: `1 - H(p) / log(num_classes)` — normalized entropy (opt-in via config)
- Raise `RuntimeError` identifying exit layer if logits contain NaN or Inf

### Req 3 — Threshold-Based Exit Decision
- Process exit heads in ascending layer order
- Return at first head where `confidence >= threshold`
- Fall back to final head if nothing fires
- Threshold is a runtime parameter — no model reload needed

### Req 4 — Joint Training
- Loss = weighted sum of cross-entropy from all exit heads + final head
- Loss weights are configurable; default is equal weighting
- Gradients flow through all exit heads simultaneously
- Supports SGD and Adam optimizers; cosine and step LR schedulers

### Req 5 — Metrics Collection
- Per sample: predicted label, ground truth, exit layer, inference time (ms), FLOPs consumed
- FLOPs = prefix sum of backbone FLOPs up to exit layer + exit head FLOPs
- Aggregate: accuracy, mean FLOPs, mean time, exit frequency per layer
- Serialize to JSON

### Req 6 — Threshold Sweep
- Evaluate over configurable threshold list (e.g. [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
- Produce trade-off table: accuracy, mean FLOPs, mean time, exit frequency per threshold
- Generate accuracy vs. FLOPs plot and accuracy vs. time plot
- Serialize table to CSV

### Req 7 — Baseline Comparison
- Always include threshold=1.0 (full-depth baseline) in the trade-off table
- Report FLOPs reduction: `(baseline_flops - early_exit_flops) / baseline_flops`
- Report accuracy drop: `baseline_accuracy - early_exit_accuracy`

### Req 8 — Configuration and Reproducibility
- Single YAML config file drives all experiment parameters
- Log resolved config to `output_dir/config.yaml` at experiment start
- Apply random seed to PyTorch, NumPy, and Python RNGs before any data loading or model init
- Raise `ConfigurationError` with field name if any required field is missing
- Config round-trip: `parse → print → parse` produces equivalent config

### Req 9 — Dataset Support
- CIFAR-10 (10 classes) and CIFAR-100 (100 classes) via torchvision
- Train: random crop + horizontal flip augmentation
- Eval: center-crop + normalize only
- Raise `DatasetError` before model init if path is invalid

---

## 4. Architecture

```
YAML Config
    │
    ├── ConfigLoader ──► ExperimentConfig
    │
    ├── DatasetLoader ──► train_loader, eval_loader
    │
    ├── build_backbone() ──► TransformerBackbone | MLPBackbone
    │                              │
    │                        EarlyExitModel
    │                         ├── ExitHead @ layer 2
    │                         ├── ExitHead @ layer 4
    │                         ├── ExitHead @ layer 6
    │                         └── Final Head
    │
    ├── Trainer ──► train all heads jointly with weighted loss
    │
    └── AnalysisPipeline
            ├── InferenceEngine (threshold sweep)
            ├── MetricsCollector (per-threshold metrics)
            └── Outputs: CSV + plots + config log
```

### Component Interfaces

**EarlyExitModel**
```python
def forward(self, x: Tensor) -> list[ExitOutput]
# Returns one ExitOutput per exit head + final head, in ascending layer order
```

**ExitHead**
```python
def forward(self, features: Tensor) -> ExitOutput
# ExitOutput: exit_layer, logits, confidence, predicted_class
```

**InferenceEngine**
```python
def infer(self, x: Tensor, threshold: float) -> InferenceResult
def infer_batch(self, data_loader: DataLoader, threshold: float) -> list[InferenceResult]
# InferenceResult: predicted_class, confidence, exit_layer, flops_consumed, inference_time_ms
```

**MetricsCollector**
```python
def record(self, result: InferenceResult, ground_truth: int) -> None
def aggregate(self) -> AggregatedMetrics
def save_json(self, path: str) -> None
```

**AnalysisPipeline**
```python
def run_sweep(self, thresholds: list[float]) -> TradeoffTable
def save_csv(self, table: TradeoffTable, path: str) -> None
def plot_accuracy_vs_flops(self, table: TradeoffTable, path: str) -> None
def plot_accuracy_vs_time(self, table: TradeoffTable, path: str) -> None
```

---

## 5. Data Models

```python
@dataclass
class ExperimentConfig:
    backbone: str                  # "transformer" | "mlp"
    exit_layer_indices: list[int]  # 1-based
    confidence_method: str         # "max_softmax" | "entropy"
    exit_loss_weights: list[float] # one per exit head + final
    optimizer: str                 # "sgd" | "adam"
    learning_rate: float
    lr_scheduler: str | None       # "cosine" | "step" | None
    num_epochs: int
    batch_size: int
    threshold_sweep: list[float]
    dataset: str                   # "cifar10" | "cifar100"
    dataset_path: str
    augmentation: AugmentationConfig
    random_seed: int
    output_dir: str

@dataclass
class LayerFLOPs:
    layer_index: int
    backbone_flops: int   # cumulative backbone FLOPs up to this layer
    exit_head_flops: int  # FLOPs for the exit head at this layer

@dataclass
class AggregatedMetrics:
    accuracy: float
    mean_flops: float
    mean_inference_time_ms: float
    exit_frequency: dict[int, float]  # exit_layer -> fraction of samples

@dataclass
class TradeoffRow:
    threshold: float
    accuracy: float
    mean_flops: float
    mean_inference_time_ms: float
    exit_frequency: dict[int, float]
    flops_reduction: float   # vs baseline
    accuracy_drop: float     # vs baseline
```

---

## 6. FLOPs Accounting

FLOPs are computed analytically at model construction time (not profiled), ensuring hardware-independent reproducibility.

**Transformer layer**: `4 * seq_len * embed_dim^2` (attention) + `2 * seq_len * embed_dim * mlp_dim` (FFN)

**MLP layer**: `2 * in_dim * out_dim` (multiply-add for linear layer)

**Exit head**: `2 * feature_dim * num_classes`

For a sample exiting at layer k: `flops = layer_flops[k].backbone_flops + layer_flops[k].exit_head_flops`

---

