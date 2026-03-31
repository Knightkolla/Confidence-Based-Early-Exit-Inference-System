import textwrap
from pathlib import Path

import pytest

from src.config.errors import ConfigurationError
from src.config.loader import load_config
from src.config.types import AugmentationConfig, ExperimentConfig

_MINIMAL_YAML = textwrap.dedent("""\
    backbone: cnn
    exit_layer_indices: [2, 4]
    confidence_method: max_softmax
    optimizer: sgd
    learning_rate: 0.1
    num_epochs: 10
    batch_size: 128
    threshold_sweep: [0.5, 0.9]
    dataset: cifar10
    dataset_path: /data/cifar10
    random_seed: 42
    output_dir: /tmp/out
""")


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return p


def test_load_minimal_config(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _MINIMAL_YAML))
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.backbone == "cnn"
    assert cfg.exit_layer_indices == [2, 4]
    assert cfg.confidence_method == "max_softmax"
    assert cfg.optimizer == "sgd"
    assert cfg.learning_rate == 0.1
    assert cfg.num_epochs == 10
    assert cfg.batch_size == 128
    assert cfg.threshold_sweep == [0.5, 0.9]
    assert cfg.dataset == "cifar10"
    assert cfg.dataset_path == "/data/cifar10"
    assert cfg.random_seed == 42
    assert cfg.output_dir == "/tmp/out"


def test_default_lr_scheduler_is_none(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _MINIMAL_YAML))
    assert cfg.lr_scheduler is None


def test_default_augmentation(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _MINIMAL_YAML))
    assert cfg.augmentation == AugmentationConfig()


def test_default_exit_loss_weights_are_equal(tmp_path: Path) -> None:
    # 2 early exits -> 3 heads total -> each weight = 1/3
    cfg = load_config(_write(tmp_path, _MINIMAL_YAML))
    assert len(cfg.exit_loss_weights) == 3
    assert all(abs(w - 1 / 3) < 1e-9 for w in cfg.exit_loss_weights)


def test_explicit_exit_loss_weights(tmp_path: Path) -> None:
    yaml = _MINIMAL_YAML + "exit_loss_weights: [0.2, 0.3, 0.5]\n"
    cfg = load_config(_write(tmp_path, yaml))
    assert cfg.exit_loss_weights == [0.2, 0.3, 0.5]


def test_explicit_lr_scheduler(tmp_path: Path) -> None:
    yaml = _MINIMAL_YAML + "lr_scheduler: cosine\n"
    cfg = load_config(_write(tmp_path, yaml))
    assert cfg.lr_scheduler == "cosine"


def test_explicit_augmentation(tmp_path: Path) -> None:
    yaml = _MINIMAL_YAML + textwrap.dedent("""\
        augmentation:
          random_crop: false
          horizontal_flip: false
          crop_padding: 8
    """)
    cfg = load_config(_write(tmp_path, yaml))
    assert cfg.augmentation.random_crop is False
    assert cfg.augmentation.horizontal_flip is False
    assert cfg.augmentation.crop_padding == 8


@pytest.mark.parametrize("missing_field", [
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
])
def test_missing_required_field_raises(tmp_path: Path, missing_field: str) -> None:
    lines = [l for l in _MINIMAL_YAML.splitlines() if not l.startswith(missing_field)]
    yaml = "\n".join(lines) + "\n"
    with pytest.raises(ConfigurationError) as exc_info:
        load_config(_write(tmp_path, yaml))
    assert missing_field in str(exc_info.value)


# --- PrettyPrinter tests ---

from src.config.pretty_printer import PrettyPrinter


def _make_config(**overrides) -> ExperimentConfig:
    defaults = dict(
        backbone="cnn",
        exit_layer_indices=[2, 4],
        confidence_method="max_softmax",
        exit_loss_weights=[1 / 3, 1 / 3, 1 / 3],
        optimizer="sgd",
        learning_rate=0.1,
        lr_scheduler=None,
        num_epochs=10,
        batch_size=128,
        threshold_sweep=[0.5, 0.9],
        dataset="cifar10",
        dataset_path="/data/cifar10",
        augmentation=AugmentationConfig(),
        random_seed=42,
        output_dir="/tmp/out",
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def test_pretty_printer_produces_valid_yaml() -> None:
    import yaml

    pp = PrettyPrinter()
    output = pp.format(_make_config())
    parsed = yaml.safe_load(output)
    assert isinstance(parsed, dict)
    assert parsed["backbone"] == "cnn"


def test_pretty_printer_round_trip(tmp_path: Path) -> None:
    cfg = _make_config()
    pp = PrettyPrinter()
    yaml_str = pp.format(cfg)
    p = tmp_path / "config.yaml"
    p.write_text(yaml_str)
    cfg2 = load_config(p)
    assert cfg == cfg2


def test_pretty_printer_round_trip_with_scheduler(tmp_path: Path) -> None:
    cfg = _make_config(lr_scheduler="cosine")
    pp = PrettyPrinter()
    p = tmp_path / "config.yaml"
    p.write_text(pp.format(cfg))
    cfg2 = load_config(p)
    assert cfg2.lr_scheduler == "cosine"


def test_pretty_printer_round_trip_custom_augmentation(tmp_path: Path) -> None:
    cfg = _make_config(augmentation=AugmentationConfig(random_crop=False, horizontal_flip=False, crop_padding=8))
    pp = PrettyPrinter()
    p = tmp_path / "config.yaml"
    p.write_text(pp.format(cfg))
    cfg2 = load_config(p)
    assert cfg2.augmentation.random_crop is False
    assert cfg2.augmentation.crop_padding == 8
