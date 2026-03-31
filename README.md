# Confidence-Based Early Exit Inference System

A research framework for studying confidence-based early exit inference in deep neural networks. The core question: **how much computation can you skip on easy inputs while preserving accuracy?**

The system attaches lightweight classifier heads at intermediate layers of a Transformer or MLP backbone. During inference, each head computes a confidence score. If confidence exceeds a threshold, the model returns that prediction immediately — skipping all remaining layers. A threshold sweep then characterizes the full accuracy vs. FLOPs trade-off curve.

---

## How It Works

```
Input image
    │
  Layer 1 ──► ExitHead 1 ──► confidence ≥ threshold? → return prediction
    │
  Layer 2 ──► ExitHead 2 ──► confidence ≥ threshold? → return prediction
    │
   ...
    │
  Layer N ──► Final Head  (always runs if nothing fired)
```

All exit heads are trained jointly with a weighted cross-entropy loss. At inference time, the threshold is a runtime parameter — no retraining needed to evaluate different operating points.

---

## Project Structure

```
src/
  config/         # YAML config loading, validation, pretty-printing
  data/           # CIFAR-10/100 dataset loading with augmentation
  models/         # TransformerBackbone, MLPBackbone, ExitHead, EarlyExitModel
  engine/         # Trainer, InferenceEngine
  metrics/        # MetricsCollector, data types
  analysis/       # Threshold sweep, trade-off table, plots
  main.py         # CLI entry point
tests/
  unit/           # Unit tests for all components
  integration/    # Full pipeline integration test
src/config.yaml   # Example experiment configuration
```

---

## Quickstart

**1. Set up the environment**

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision pyyaml matplotlib hypothesis tqdm
```

**2. Download CIFAR-10**

```bash
python -c "import torchvision; torchvision.datasets.CIFAR10('./data', download=True)"
```

**3. Configure your experiment**

Edit `src/config.yaml`:

```yaml
backbone: transformer        # "transformer" or "mlp"
exit_layer_indices: [2, 4, 6]
confidence_method: max_softmax
optimizer: adam
learning_rate: 0.001
lr_scheduler: cosine
num_epochs: 50
batch_size: 128
threshold_sweep: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
dataset: cifar10
dataset_path: ./data
random_seed: 42
output_dir: ./outputs
```

**4. Train and evaluate**

```bash
python -m src.main --config src/config.yaml
```

Training logs to the terminal per epoch:

```
Epoch   1/50: 100%|████████| 391/391 [00:22<00:00, loss=2.1843]
Epoch   1/50  loss=2.1843  val_acc=0.1823  lr=0.001000
Epoch   2/50: 100%|████████| 391/391 [00:21<00:00, loss=1.8901]
Epoch   2/50  loss=1.8901  val_acc=0.2541  lr=0.000994
```

**5. View results**

```bash
open outputs/accuracy_vs_flops.png
open outputs/accuracy_vs_time.png
cat outputs/tradeoff_table.csv
```

---

## Outputs

| File | Description |
|---|---|
| `outputs/config.yaml` | Resolved config logged for reproducibility |
| `outputs/tradeoff_table.csv` | Accuracy, FLOPs, time, exit frequency per threshold |
| `outputs/accuracy_vs_flops.png` | Trade-off curve: accuracy vs. mean FLOPs |
| `outputs/accuracy_vs_time.png` | Trade-off curve: accuracy vs. mean inference time |

---

## Configuration Reference

| Field | Type | Description |
|---|---|---|
| `backbone` | str | `"transformer"` or `"mlp"` |
| `exit_layer_indices` | list[int] | 1-based layer indices for exit heads |
| `confidence_method` | str | `"max_softmax"` or `"entropy"` |
| `exit_loss_weights` | list[float] | Per-head loss weights (defaults to equal) |
| `optimizer` | str | `"adam"` or `"sgd"` |
| `learning_rate` | float | Initial learning rate |
| `lr_scheduler` | str or null | `"cosine"`, `"step"`, or `null` |
| `num_epochs` | int | Training epochs |
| `batch_size` | int | Batch size |
| `threshold_sweep` | list[float] | Confidence thresholds to evaluate |
| `dataset` | str | `"cifar10"` or `"cifar100"` |
| `dataset_path` | str | Path to dataset root |
| `random_seed` | int | Seed for full reproducibility |
| `output_dir` | str | Directory for all outputs |

---

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/unit/ -v

# Integration test
python -m pytest tests/integration/ -v
```

---

## Hardware Notes

- Automatically uses MPS (Apple Silicon), CUDA, or CPU — detected at runtime
- Transformer backbone on M2 (10-core GPU): ~20-30s per epoch on CIFAR-10
- For a quick smoke run, set `num_epochs: 2` and `backbone: mlp`
