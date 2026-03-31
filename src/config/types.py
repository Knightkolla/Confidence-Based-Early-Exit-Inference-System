from dataclasses import dataclass, field


@dataclass
class AugmentationConfig:
    random_crop: bool = True
    horizontal_flip: bool = True
    crop_padding: int = 4


@dataclass
class ExperimentConfig:
    # Model
    backbone: str                    # "cnn" | "transformer" | "mlp"
    exit_layer_indices: list[int]    # 1-based, must be < total layers
    confidence_method: str           # "max_softmax" | "entropy"
    exit_loss_weights: list[float]   # one per exit head + final; defaults to equal

    # Training
    optimizer: str                   # "sgd" | "adam"
    learning_rate: float
    lr_scheduler: str | None         # "cosine" | "step" | None
    num_epochs: int
    batch_size: int

    # Evaluation
    threshold_sweep: list[float]     # e.g. [0.5, 0.6, ..., 0.99]

    # Data
    dataset: str                     # "cifar10" | "cifar100"
    dataset_path: str
    augmentation: AugmentationConfig

    # Reproducibility
    random_seed: int

    # Output
    output_dir: str
