# Usage: python -m src.main --config path/to/config.yaml

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from src.config.errors import ConfigurationError
from src.config.loader import load_config
from src.data.loader import DatasetLoader
from src.engine.trainer import Trainer
from src.analysis.pipeline import AnalysisPipeline
from src.models.early_exit_model import EarlyExitModel
from src.models.mlp_backbone import MLPBackbone
from src.models.transformer_backbone import TransformerBackbone

_DATASET_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_backbone(config, num_classes: int) -> torch.nn.Module:
    if config.backbone == "transformer":
        return TransformerBackbone(
            num_layers=6,
            num_classes=num_classes,
            image_size=32,
            patch_size=4,
            embed_dim=256,
            num_heads=8,
        )
    if config.backbone == "mlp":
        return MLPBackbone(
            num_layers=6,
            num_classes=num_classes,
            image_size=32,
            hidden_dim=512,
        )
    raise ConfigurationError(
        f"CNN backbone not yet implemented; use 'transformer' or 'mlp'"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confidence-based early exit training and evaluation"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)

    seed_everything(config.random_seed)

    num_classes = _DATASET_NUM_CLASSES.get(config.dataset)
    if num_classes is None:
        raise ConfigurationError(
            f"Unsupported dataset '{config.dataset}'. Valid options: {sorted(_DATASET_NUM_CLASSES)}"
        )

    dataset_loader = DatasetLoader()
    train_loader, eval_loader = dataset_loader.load(
        dataset=config.dataset,
        path=config.dataset_path,
        augmentation=config.augmentation,
        batch_size=config.batch_size,
    )

    device = get_device()
    print(f"Using device: {device}")

    backbone = build_backbone(config, num_classes)
    model = EarlyExitModel(
        backbone=backbone,
        exit_layer_indices=config.exit_layer_indices,
        num_classes=num_classes,
        confidence_method=config.confidence_method,
    )
    model = model.to(device)

    trainer = Trainer(model, config, device=device)
    trainer.train(train_loader, eval_loader=eval_loader)

    pipeline = AnalysisPipeline(model, eval_loader, device=device)
    table = pipeline.run_sweep(config.threshold_sweep)

    output_dir = Path(config.output_dir)
    pipeline.save_csv(table, str(output_dir / "tradeoff_table.csv"))
    pipeline.plot_accuracy_vs_flops(table, str(output_dir / "accuracy_vs_flops.png"))
    pipeline.plot_accuracy_vs_time(table, str(output_dir / "accuracy_vs_time.png"))


if __name__ == "__main__":
    main()
