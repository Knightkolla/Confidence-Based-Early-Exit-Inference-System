# Implementation Plan: Confidence-Based Early Exit

## Overview

Implement the full confidence-based early exit system using a Transformer/MLP backbone (Option B) on CIFAR-10/100. Tasks proceed incrementally: core data models and config → dataset loading → model construction → training → inference engine → metrics collection → analysis pipeline → integration wiring.

## Tasks

- [x] 1. Project structure and core data models
  - Create `src/` package with `__init__.py` files for: `models/`, `engine/`, `metrics/`, `analysis/`, `config/`, `data/`
  - Implement `ExitOutput` and `InferenceResult` dataclasses in `src/models/types.py`
  - Implement `AggregatedMetrics`, `LayerFLOPs`, and `TradeoffTable` dataclasses in `src/metrics/types.py`
  - Implement `ExperimentConfig` and `AugmentationConfig` dataclasses in `src/config/types.py`
  - _Requirements: 1.4, 5.1, 5.2, 8.1_

- [x] 2. Configuration system
  - [x] 2.1 Implement `ConfigLoader` in `src/config/loader.py`
    - Parse YAML with `PyYAML`; validate all required fields; raise `ConfigurationError` with field name on missing fields
    - Apply defaults for optional fields (equal loss weights, no scheduler)
    - _Requirements: 8.1, 8.4_
  - [ ]* 2.2 Write property test for missing required config field raises before computation (Property 19)
    - **Property 19: Missing required config field raises before computation**
    - **Validates: Requirements 8.4**
  - [x] 2.3 Implement `PrettyPrinter` in `src/config/pretty_printer.py`
    - Serialize `ExperimentConfig` back to valid YAML string
    - _Requirements: 8.5_
  - [ ]* 2.4 Write property test for configuration round-trip (Property 20)
    - **Property 20: Configuration round-trip**
    - **Validates: Requirements 8.5, 8.6**
  - [x] 2.5 Write unit tests for `ConfigLoader` and `PrettyPrinter` in `tests/unit/test_config_loader.py`
    - Test config log file written to output directory (Req 8.2)
    - Test valid YAML parses to correct `ExperimentConfig`
    - _Requirements: 8.1, 8.2_

- [x] 3. Dataset loading
  - [x] 3.1 Implement `DatasetLoader` in `src/data/loader.py`
    - Support CIFAR-10 and CIFAR-100 via `torchvision.datasets`
    - Apply augmentation transforms (random crop, horizontal flip) for train split; center-crop + normalize for eval
    - Raise `DatasetError` before model init if path invalid
    - _Requirements: 9.1, 9.2, 9.3, 9.4_
  - [ ]* 3.2 Write property test for training vs eval transforms (Property 21)
    - **Property 21: Training transforms include augmentation; eval transforms do not**
    - **Validates: Requirements 9.3**
  - [ ]* 3.3 Write property test for invalid dataset path raises before model initialization (Property 22)
    - **Property 22: Invalid dataset path raises before model initialization**
    - **Validates: Requirements 9.4**
  - [x] 3.4 Write unit tests for `DatasetLoader` in `tests/unit/test_dataset_loader.py`
    - Test CIFAR-10 loads with correct train/test split sizes (Req 9.1)
    - Test CIFAR-100 loads for Transformer backbone (Req 9.2)
    - _Requirements: 9.1, 9.2_

- [x] 4. ExitHead and EarlyExitModel construction
  - [x] 4.1 Implement `ExitHead` in `src/models/exit_head.py`
    - Linear classifier head; compute `max_softmax` and `entropy` confidence modes
    - Raise `RuntimeError` with exit layer index if logits contain NaN/Inf
    - Return `ExitOutput` dataclass
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  - [ ]* 4.2 Write property test for max-softmax confidence correctness (Property 3)
    - **Property 3: Max-softmax confidence correctness**
    - **Validates: Requirements 2.1, 2.2**
  - [ ]* 4.3 Write property test for entropy-based confidence correctness (Property 4)
    - **Property 4: Entropy-based confidence correctness**
    - **Validates: Requirements 2.3**
  - [ ]* 4.4 Write property test for non-finite logits raise a runtime error (Property 5)
    - **Property 5: Non-finite logits raise a runtime error**
    - **Validates: Requirements 2.4**
  - [x] 4.5 Implement Transformer backbone in `src/models/transformer_backbone.py`
    - Patch embedding + N transformer encoder blocks (configurable depth) for CIFAR-10/100
    - Expose `layers` as `nn.ModuleList` so exit indices can be validated
    - Compute and store `LayerFLOPs` analytically at construction time
    - _Requirements: 1.3, 5.2_
  - [x] 4.6 Implement MLP backbone in `src/models/mlp_backbone.py`
    - Configurable depth MLP; expose `layers` as `nn.ModuleList`
    - Compute and store `LayerFLOPs` analytically
    - _Requirements: 1.3, 5.2_
  - [x] 4.7 Implement `EarlyExitModel` in `src/models/early_exit_model.py`
    - Accept backbone + `exit_layer_indices` + `confidence_method`; attach `ExitHead` at each index
    - Validate indices at `__init__`; raise `ConfigurationError` if any index exceeds backbone depth
    - `forward()` runs all layers and returns `list[ExitOutput]`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [ ]* 4.8 Write property test for exit heads placed at configured indices (Property 1)
    - **Property 1: Exit heads are placed at configured indices**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
  - [ ]* 4.9 Write property test for invalid exit layer index raises at construction (Property 2)
    - **Property 2: Invalid exit layer index raises at construction**
    - **Validates: Requirements 1.5**
  - [x] 4.10 Write unit tests for `ExitHead` and `EarlyExitModel` in `tests/unit/test_exit_head.py`
    - Test both confidence modes return values in [0, 1]
    - Test `ConfigurationError` on invalid index
    - _Requirements: 1.5, 2.1, 2.2, 2.3, 2.4_

- [x] 5. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Training procedure
  - [x] 6.1 Implement `Trainer` in `src/engine/trainer.py`
    - Compute weighted cross-entropy loss across all exit heads + final head
    - Support SGD and Adam optimizers; cosine and step LR schedulers
    - Apply random seed to PyTorch and NumPy before data loading and model init
    - Log resolved config to `output_dir/config.yaml` at experiment start
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 8.2, 8.3_
  - [ ]* 6.2 Write property test for weighted loss equals sum of per-exit losses (Property 8)
    - **Property 8: Weighted loss equals sum of per-exit losses**
    - **Validates: Requirements 4.1, 4.2**
  - [ ]* 6.3 Write property test for all exit head parameters receive gradients (Property 9)
    - **Property 9: All exit head parameters receive gradients**
    - **Validates: Requirements 4.3**
  - [ ]* 6.4 Write property test for seeded runs are reproducible (Property 18)
    - **Property 18: Seeded runs are reproducible**
    - **Validates: Requirements 8.3**
  - [x] 6.5 Write unit tests for `Trainer` in `tests/unit/test_trainer.py`
    - Test SGD and Adam train for 1 step without error (Req 4.4)
    - Test cosine and step schedulers step correctly
    - _Requirements: 4.4_

- [x] 7. Inference engine
  - [x] 7.1 Implement `InferenceEngine` in `src/engine/inference.py`
    - Process exit heads in ascending layer order; return at first `confidence >= threshold`
    - Fall back to final head if no exit fires
    - Accept threshold as runtime parameter (no model reload)
    - Return `InferenceResult` with `flops_consumed` from prefix-sum `LayerFLOPs`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 5.2_
  - [ ]* 7.2 Write property test for early exit fires at first confident head (Property 6)
    - **Property 6: Early exit fires at first confident head**
    - **Validates: Requirements 3.1, 3.4**
  - [ ]* 7.3 Write property test for fallback to final layer when no exit fires (Property 7)
    - **Property 7: Fallback to final layer when no exit fires**
    - **Validates: Requirements 3.2**
  - [ ]* 7.4 Write property test for FLOPs per sample equals prefix sum up to exit layer (Property 11)
    - **Property 11: FLOPs per sample equals prefix sum up to exit layer**
    - **Validates: Requirements 5.2**
  - [ ]* 7.5 Write property test for threshold 1.0 forces all samples to final layer (Property 16)
    - **Property 16: Threshold 1.0 forces all samples to final layer**
    - **Validates: Requirements 7.1**
  - [x] 7.6 Write unit tests for `InferenceEngine` in `tests/unit/test_inference_engine.py`
    - Test threshold boundary (exactly equal fires, just below does not)
    - Test fallback path
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 8. Metrics collection
  - [x] 8.1 Implement `MetricsCollector` in `src/metrics/collector.py`
    - `record()` stores predicted label, ground-truth, exit layer, inference time ms
    - `aggregate()` computes accuracy, mean FLOPs, mean time, exit frequency dict
    - `save_json()` serializes `AggregatedMetrics` to JSON
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  - [ ]* 8.2 Write property test for MetricsCollector records all required fields per sample (Property 10)
    - **Property 10: MetricsCollector records all required fields per sample**
    - **Validates: Requirements 5.1**
  - [ ]* 8.3 Write property test for exit frequency sums to 1 (Property 12)
    - **Property 12: Exit frequency sums to 1**
    - **Validates: Requirements 5.3, 5.4**
  - [ ]* 8.4 Write property test for metrics JSON round-trip (Property 13)
    - **Property 13: Metrics JSON round-trip**
    - **Validates: Requirements 5.5**
  - [x] 8.5 Write unit tests for `MetricsCollector` in `tests/unit/test_metrics_collector.py`
    - Test `aggregate()` on known sample set produces correct accuracy and exit frequencies
    - _Requirements: 5.3, 5.4_

- [x] 9. Analysis pipeline
  - [x] 9.1 Implement `AnalysisPipeline` in `src/analysis/pipeline.py`
    - `run_sweep()` iterates threshold list, calls `InferenceEngine.infer()` for each, collects `MetricsCollector` output
    - Include threshold=1.0 baseline row; compute FLOPs reduction and accuracy drop vs baseline
    - `save_csv()` serializes `TradeoffTable` to CSV
    - `plot_accuracy_vs_flops()` and `plot_accuracy_vs_time()` save plots via `matplotlib`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3_
  - [ ]* 9.2 Write property test for trade-off table completeness (Property 14)
    - **Property 14: Trade-off table completeness**
    - **Validates: Requirements 6.1, 6.2**
  - [ ]* 9.3 Write property test for trade-off CSV round-trip (Property 15)
    - **Property 15: Trade-off CSV round-trip**
    - **Validates: Requirements 6.5**
  - [ ]* 9.4 Write property test for comparison metrics are computed correctly (Property 17)
    - **Property 17: Comparison metrics are computed correctly**
    - **Validates: Requirements 7.2, 7.3**
  - [x] 9.5 Write unit tests for `AnalysisPipeline` in `tests/unit/test_analysis_pipeline.py`
    - Test plots are generated at configured paths (Req 6.3, 6.4)
    - Test baseline row always present in trade-off table
    - _Requirements: 6.3, 6.4, 7.1_

- [x] 10. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [-] 11. Wire everything together
  - [-] 11.1 Implement `src/main.py` entry point
    - Parse CLI arg `--config`; call `ConfigLoader.load()`; seed RNGs; build backbone + `EarlyExitModel`; run `Trainer`; run `AnalysisPipeline.run_sweep()`; write outputs
    - _Requirements: 8.1, 8.2, 8.3_
  - [~] 11.2 Write integration test in `tests/integration/test_full_pipeline.py`
    - Train Transformer backbone for 1 epoch on a tiny CIFAR-10 subset; run threshold sweep; assert output files (CSV, plots, JSON, config log) exist
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 8.2_

- [~] 12. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Property tests use `hypothesis` with `@settings(max_examples=100)`; each is tagged `# Feature: confidence-based-early-exit, Property N: <text>`
- All 22 correctness properties from the design document are covered by property sub-tasks
- Backbone priority is Transformer (Option B); MLP is a secondary variant sharing the same interfaces
