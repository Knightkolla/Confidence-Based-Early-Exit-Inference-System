# Requirements Document

## Introduction

A confidence-based early exit inference system that allows deep neural networks to terminate processing before completing all layers when sufficient prediction confidence is achieved. The system targets the research question: "How effectively can confidence-based early stopping reduce computation while preserving accuracy in deep models?"

The system supports two implementation tracks:
- Option A: CNN on CIFAR-10 with early exit layers (simplest, fast baseline)
- Option B: Transformer/MLP with progressive inference and adaptive depth (stronger variant)

Both tracks share the same early exit framework, metrics collection, and analysis pipeline.

## Glossary

- **EarlyExitModel**: The neural network model augmented with intermediate exit points at multiple layers
- **ExitHead**: A lightweight classifier attached to an intermediate layer that produces a confidence score and prediction
- **ConfidenceScore**: A scalar value in [0, 1] representing the model's certainty at a given exit point, computed via softmax entropy or max-softmax probability
- **ExitThreshold**: A configurable scalar in (0, 1] above which the system accepts an ExitHead's prediction and halts further processing
- **InferenceEngine**: The runtime component that feeds inputs through the EarlyExitModel and applies exit decisions
- **MetricsCollector**: The component that records accuracy, FLOPs, inference time, and exit frequency per sample
- **AnalysisPipeline**: The component that aggregates MetricsCollector output and produces trade-off reports
- **FLOPs**: Floating point operations, used as a proxy for computational cost
- **ExitLayer**: The index (1-based) of the layer at which an early exit decision is made
- **Backbone**: The base neural network architecture (CNN or Transformer/MLP) without exit heads

---

## Requirements

### Requirement 1: Early Exit Model Construction

**User Story:** As a researcher, I want to attach exit heads to intermediate layers of a deep model, so that the model can produce predictions at multiple depths.

#### Acceptance Criteria

1. THE EarlyExitModel SHALL attach at least two ExitHeads to distinct intermediate layers of the Backbone.
2. WHEN the Backbone is a CNN, THE EarlyExitModel SHALL place ExitHeads after convolutional blocks at configurable layer indices.
3. WHEN the Backbone is a Transformer or MLP, THE EarlyExitModel SHALL place ExitHeads after configurable transformer/MLP blocks.
4. THE EarlyExitModel SHALL expose a configuration parameter specifying the list of ExitLayer indices.
5. IF a specified ExitLayer index exceeds the total number of Backbone layers, THEN THE EarlyExitModel SHALL raise a descriptive configuration error at initialization time.

---

### Requirement 2: Confidence Score Computation

**User Story:** As a researcher, I want each exit head to compute a confidence score, so that the system can decide whether to stop or continue processing.

#### Acceptance Criteria

1. WHEN an ExitHead produces logits for an input, THE ExitHead SHALL compute a ConfidenceScore as the maximum softmax probability over all classes.
2. THE ExitHead SHALL return both the predicted class label and the ConfidenceScore for each input.
3. THE EarlyExitModel SHALL support an alternative ConfidenceScore based on normalized entropy: `1 - H(p) / log(num_classes)`, selectable via configuration.
4. IF the logits produced by an ExitHead contain non-finite values (NaN or Inf), THEN THE ExitHead SHALL raise a runtime error identifying the ExitLayer index.

---

### Requirement 3: Threshold-Based Exit Decision

**User Story:** As a researcher, I want the inference engine to stop processing when confidence exceeds a threshold, so that computation is saved on easy inputs.

#### Acceptance Criteria

1. WHEN the InferenceEngine evaluates an ExitHead and the ConfidenceScore is greater than or equal to the ExitThreshold, THE InferenceEngine SHALL return the ExitHead's prediction and record the ExitLayer index.
2. WHEN the InferenceEngine evaluates all ExitHeads and no ConfidenceScore meets the ExitThreshold, THE InferenceEngine SHALL return the final Backbone layer's prediction.
3. THE InferenceEngine SHALL accept the ExitThreshold as a runtime parameter, allowing evaluation across multiple threshold values without reloading the model.
4. THE InferenceEngine SHALL process ExitHeads in ascending ExitLayer order, stopping at the first exit that meets the threshold.

---

### Requirement 4: Training with Early Exit Heads

**User Story:** As a researcher, I want to train the model including all exit heads jointly, so that each exit head learns to produce accurate predictions.

#### Acceptance Criteria

1. WHEN training the EarlyExitModel, THE training procedure SHALL compute a weighted sum of cross-entropy losses from all ExitHeads and the final Backbone output.
2. THE training procedure SHALL expose a configurable loss weight per ExitHead, defaulting to equal weighting across all exits.
3. WHEN a training batch is processed, THE EarlyExitModel SHALL compute gradients through all ExitHeads simultaneously.
4. THE training procedure SHALL support standard optimizers (SGD, Adam) and learning rate schedulers without modification to the exit head architecture.

---

### Requirement 5: Metrics Collection

**User Story:** As a researcher, I want the system to record accuracy, FLOPs, inference time, and exit frequency, so that I can analyze the trade-offs.

#### Acceptance Criteria

1. WHEN the InferenceEngine processes a sample, THE MetricsCollector SHALL record the predicted label, ground-truth label, ExitLayer index, and wall-clock inference time in milliseconds.
2. THE MetricsCollector SHALL compute the FLOPs consumed per sample as the sum of FLOPs of all Backbone layers up to and including the ExitLayer.
3. THE MetricsCollector SHALL record exit frequency as the fraction of samples exiting at each ExitLayer index across a full evaluation dataset.
4. WHEN evaluation is complete, THE MetricsCollector SHALL compute and expose overall accuracy, mean FLOPs per sample, mean inference time per sample, and per-exit-layer exit frequency.
5. THE MetricsCollector SHALL serialize collected metrics to a JSON file at a configurable output path.

---

### Requirement 6: Threshold Sweep and Trade-off Analysis

**User Story:** As a researcher, I want to evaluate the model across a range of confidence thresholds, so that I can characterize the accuracy vs. computation trade-off curve.

#### Acceptance Criteria

1. THE AnalysisPipeline SHALL evaluate the EarlyExitModel over a configurable list of ExitThreshold values (e.g., [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]).
2. WHEN the threshold sweep is complete, THE AnalysisPipeline SHALL produce a trade-off table containing, for each threshold: accuracy, mean FLOPs, mean inference time, and exit frequency per layer.
3. THE AnalysisPipeline SHALL generate a plot of accuracy vs. mean FLOPs across all evaluated thresholds.
4. THE AnalysisPipeline SHALL generate a plot of accuracy vs. mean inference time across all evaluated thresholds.
5. THE AnalysisPipeline SHALL serialize the trade-off table to a CSV file at a configurable output path.

---

### Requirement 7: Baseline Comparison

**User Story:** As a researcher, I want to compare the early exit model against a full-depth baseline, so that I can quantify the computational savings.

#### Acceptance Criteria

1. THE AnalysisPipeline SHALL evaluate a full-depth baseline (ExitThreshold = 1.0, all samples reach the final layer) and include its metrics in the trade-off table.
2. WHEN comparing early exit results to the baseline, THE AnalysisPipeline SHALL compute and report FLOPs reduction as `(baseline_FLOPs - early_exit_FLOPs) / baseline_FLOPs`.
3. WHEN comparing early exit results to the baseline, THE AnalysisPipeline SHALL compute and report accuracy drop as `baseline_accuracy - early_exit_accuracy`.

---

### Requirement 8: Configuration and Reproducibility

**User Story:** As a researcher, I want all experiment parameters to be specified in a single configuration file, so that experiments are reproducible.

#### Acceptance Criteria

1. THE system SHALL accept a YAML configuration file specifying: Backbone architecture, ExitLayer indices, ExitThreshold sweep values, loss weights, optimizer settings, random seed, dataset path, and output directory.
2. WHEN an experiment is run, THE system SHALL log the resolved configuration (including defaults) to a text file in the output directory.
3. THE system SHALL accept a random seed parameter and apply it to all random number generators (PyTorch, NumPy) before any data loading or model initialization.
4. IF the configuration file is missing a required field, THEN THE system SHALL raise a descriptive error identifying the missing field before any computation begins.
5. THE Pretty_Printer SHALL format the resolved configuration back into a valid YAML string.
6. FOR ALL valid configuration objects, parsing then printing then parsing SHALL produce an equivalent configuration object (round-trip property).

---

### Requirement 9: Dataset Support

**User Story:** As a researcher, I want the system to support standard benchmark datasets, so that results are comparable to published work.

#### Acceptance Criteria

1. THE system SHALL support CIFAR-10 as a dataset, loading it via standard train/test splits.
2. WHERE the Backbone is a Transformer or MLP, THE system SHALL support CIFAR-100 as an additional dataset option.
3. WHEN a dataset is loaded, THE system SHALL apply configurable data augmentation (random crop, horizontal flip) during training and center-crop normalization during evaluation.
4. IF the dataset path does not exist or the dataset cannot be downloaded, THEN THE system SHALL raise a descriptive error before model initialization.
