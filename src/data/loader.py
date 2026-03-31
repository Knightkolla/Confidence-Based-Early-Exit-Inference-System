import os
from typing import Callable

import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100

from src.config.types import AugmentationConfig
from src.data.errors import DatasetError

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)
_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)

_SUPPORTED_DATASETS = {"cifar10", "cifar100"}


def _build_train_transforms(augmentation: AugmentationConfig, mean: tuple, std: tuple) -> T.Compose:
    transforms: list[Callable] = []
    if augmentation.random_crop:
        transforms.append(T.RandomCrop(32, padding=augmentation.crop_padding))
    if augmentation.horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean, std))
    return T.Compose(transforms)


def _build_eval_transforms(mean: tuple, std: tuple) -> T.Compose:
    return T.Compose([
        T.CenterCrop(32),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class DatasetLoader:
    def load(
        self,
        dataset: str,
        path: str,
        augmentation: AugmentationConfig,
        batch_size: int,
        num_workers: int = 2,
    ) -> tuple[DataLoader, DataLoader]:
        if dataset not in _SUPPORTED_DATASETS:
            raise DatasetError(
                f"Unsupported dataset '{dataset}'. Valid options: {sorted(_SUPPORTED_DATASETS)}"
            )
        if not os.path.exists(path):
            raise DatasetError(f"Dataset path does not exist: '{path}'")

        if dataset == "cifar10":
            mean, std = _CIFAR10_MEAN, _CIFAR10_STD
            cls = CIFAR10
        else:
            mean, std = _CIFAR100_MEAN, _CIFAR100_STD
            cls = CIFAR100

        train_transform = _build_train_transforms(augmentation, mean, std)
        eval_transform = _build_eval_transforms(mean, std)

        try:
            train_dataset = cls(root=path, train=True, download=False, transform=train_transform)
            eval_dataset = cls(root=path, train=False, download=False, transform=eval_transform)
        except Exception as exc:
            raise DatasetError(f"Failed to load dataset '{dataset}' from '{path}': {exc}") from exc

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
        return train_loader, eval_loader
