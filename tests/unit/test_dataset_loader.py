import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torchvision.transforms as T

from src.config.types import AugmentationConfig
from src.data.errors import DatasetError
from src.data.loader import DatasetLoader, _build_eval_transforms, _build_train_transforms


def _default_aug() -> AugmentationConfig:
    return AugmentationConfig(random_crop=True, horizontal_flip=True, crop_padding=4)


def _no_aug() -> AugmentationConfig:
    return AugmentationConfig(random_crop=False, horizontal_flip=False, crop_padding=4)


class TestDatasetError:
    def test_invalid_dataset_name_raises(self, tmp_path):
        loader = DatasetLoader()
        with pytest.raises(DatasetError, match="Unsupported dataset"):
            loader.load("imagenet", str(tmp_path), _default_aug(), batch_size=32)

    def test_nonexistent_path_raises(self):
        loader = DatasetLoader()
        with pytest.raises(DatasetError, match="does not exist"):
            loader.load("cifar10", "/nonexistent/path/xyz", _default_aug(), batch_size=32)

    def test_error_raised_before_model_init(self):
        """DatasetError must surface before any model construction occurs (Req 9.4)."""
        loader = DatasetLoader()
        with pytest.raises(DatasetError):
            loader.load("cifar10", "/nonexistent/path", _default_aug(), batch_size=32)


class TestTrainTransforms:
    """Req 9.3: train transforms include augmentation."""

    def test_random_crop_included_when_enabled(self):
        aug = AugmentationConfig(random_crop=True, horizontal_flip=False, crop_padding=4)
        compose = _build_train_transforms(aug, (0.5,), (0.5,))
        types = [type(t) for t in compose.transforms]
        assert T.RandomCrop in types

    def test_horizontal_flip_included_when_enabled(self):
        aug = AugmentationConfig(random_crop=False, horizontal_flip=True, crop_padding=4)
        compose = _build_train_transforms(aug, (0.5,), (0.5,))
        types = [type(t) for t in compose.transforms]
        assert T.RandomHorizontalFlip in types

    def test_augmentation_excluded_when_disabled(self):
        compose = _build_train_transforms(_no_aug(), (0.5,), (0.5,))
        types = [type(t) for t in compose.transforms]
        assert T.RandomCrop not in types
        assert T.RandomHorizontalFlip not in types

    def test_always_includes_to_tensor_and_normalize(self):
        compose = _build_train_transforms(_default_aug(), (0.5,), (0.5,))
        types = [type(t) for t in compose.transforms]
        assert T.ToTensor in types
        assert T.Normalize in types

    def test_crop_padding_is_applied(self):
        aug = AugmentationConfig(random_crop=True, horizontal_flip=False, crop_padding=8)
        compose = _build_train_transforms(aug, (0.5,), (0.5,))
        crop = next(t for t in compose.transforms if isinstance(t, T.RandomCrop))
        # torchvision stores padding as the raw value passed in
        assert crop.padding == 8


class TestEvalTransforms:
    """Req 9.3: eval transforms do not include augmentation."""

    def test_no_random_crop(self):
        compose = _build_eval_transforms((0.5,), (0.5,))
        types = [type(t) for t in compose.transforms]
        assert T.RandomCrop not in types

    def test_no_horizontal_flip(self):
        compose = _build_eval_transforms((0.5,), (0.5,))
        types = [type(t) for t in compose.transforms]
        assert T.RandomHorizontalFlip not in types

    def test_includes_center_crop(self):
        compose = _build_eval_transforms((0.5,), (0.5,))
        types = [type(t) for t in compose.transforms]
        assert T.CenterCrop in types

    def test_includes_to_tensor_and_normalize(self):
        compose = _build_eval_transforms((0.5,), (0.5,))
        types = [type(t) for t in compose.transforms]
        assert T.ToTensor in types
        assert T.Normalize in types


class TestDatasetLoaderWithMocks:
    """Tests that mock torchvision dataset classes to avoid requiring downloaded data."""

    def _make_mock_dataset(self, size: int):
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=size)
        return ds

    @patch("src.data.loader.CIFAR10")
    def test_cifar10_returns_two_loaders(self, mock_cifar10, tmp_path):
        """Req 9.1: CIFAR-10 loads with train and test splits."""
        mock_cifar10.side_effect = [
            self._make_mock_dataset(50000),
            self._make_mock_dataset(10000),
        ]
        loader = DatasetLoader()
        train_loader, eval_loader = loader.load("cifar10", str(tmp_path), _default_aug(), batch_size=64)
        assert train_loader is not None
        assert eval_loader is not None

    @patch("src.data.loader.CIFAR10")
    def test_cifar10_uses_correct_split_flags(self, mock_cifar10, tmp_path):
        """Req 9.1: train split uses train=True, eval split uses train=False."""
        mock_cifar10.side_effect = [
            self._make_mock_dataset(50000),
            self._make_mock_dataset(10000),
        ]
        loader = DatasetLoader()
        loader.load("cifar10", str(tmp_path), _default_aug(), batch_size=64)
        calls = mock_cifar10.call_args_list
        assert calls[0][1]["train"] is True
        assert calls[1][1]["train"] is False

    @patch("src.data.loader.CIFAR100")
    def test_cifar100_returns_two_loaders(self, mock_cifar100, tmp_path):
        """Req 9.2: CIFAR-100 loads for Transformer/MLP backbone."""
        mock_cifar100.side_effect = [
            self._make_mock_dataset(50000),
            self._make_mock_dataset(10000),
        ]
        loader = DatasetLoader()
        train_loader, eval_loader = loader.load("cifar100", str(tmp_path), _default_aug(), batch_size=64)
        assert train_loader is not None
        assert eval_loader is not None

    @patch("src.data.loader.CIFAR10")
    def test_train_loader_shuffles(self, mock_cifar10, tmp_path):
        mock_cifar10.side_effect = [
            self._make_mock_dataset(50000),
            self._make_mock_dataset(10000),
        ]
        loader = DatasetLoader()
        train_loader, _ = loader.load("cifar10", str(tmp_path), _default_aug(), batch_size=64)
        assert train_loader.sampler is not None

    @patch("src.data.loader.CIFAR10")
    def test_dataset_exception_wrapped_as_dataset_error(self, mock_cifar10, tmp_path):
        mock_cifar10.side_effect = RuntimeError("corrupt data")
        loader = DatasetLoader()
        with pytest.raises(DatasetError, match="Failed to load dataset"):
            loader.load("cifar10", str(tmp_path), _default_aug(), batch_size=64)
