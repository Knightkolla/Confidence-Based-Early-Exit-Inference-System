from pathlib import Path

import yaml

from src.config.errors import ConfigurationError
from src.config.types import AugmentationConfig, ExperimentConfig

_REQUIRED_FIELDS = [
    "backbone",
    "exit_layer_indices",
    "confidence_method",
    "optimizer",
    "learning_rate",
    "num_epochs",
    "batch_size",
    "threshold_sweep",
    "dataset",
    "dataset_path",
    "random_seed",
    "output_dir",
]


def _parse_augmentation(raw: dict) -> AugmentationConfig:
    return AugmentationConfig(
        random_crop=raw.get("random_crop", True),
        horizontal_flip=raw.get("horizontal_flip", True),
        crop_padding=raw.get("crop_padding", 4),
    )


def _equal_weights(num_exits: int) -> list[float]:
    # One weight per early-exit head plus the final classifier head.
    n = num_exits + 1
    return [1.0 / n] * n


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r") as f:
        raw: dict = yaml.safe_load(f) or {}

    for field in _REQUIRED_FIELDS:
        if field not in raw:
            raise ConfigurationError(field)

    exit_layer_indices: list[int] = raw["exit_layer_indices"]

    exit_loss_weights: list[float] = raw.get(
        "exit_loss_weights", _equal_weights(len(exit_layer_indices))
    )

    augmentation = _parse_augmentation(raw.get("augmentation", {}))

    return ExperimentConfig(
        backbone=raw["backbone"],
        exit_layer_indices=exit_layer_indices,
        confidence_method=raw["confidence_method"],
        exit_loss_weights=exit_loss_weights,
        optimizer=raw["optimizer"],
        learning_rate=raw["learning_rate"],
        lr_scheduler=raw.get("lr_scheduler", None),
        num_epochs=raw["num_epochs"],
        batch_size=raw["batch_size"],
        threshold_sweep=raw["threshold_sweep"],
        dataset=raw["dataset"],
        dataset_path=raw["dataset_path"],
        augmentation=augmentation,
        random_seed=raw["random_seed"],
        output_dir=raw["output_dir"],
    )
