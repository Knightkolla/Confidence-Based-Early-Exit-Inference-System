from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.pretty_printer import PrettyPrinter
from src.config.types import ExperimentConfig
from src.models.early_exit_model import EarlyExitModel


class Trainer:
    def __init__(
        self,
        model: EarlyExitModel,
        config: ExperimentConfig,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device or torch.device("cpu")

        num_epochs = config.num_epochs

        if config.optimizer == "sgd":
            self.optimizer = SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=1e-4,
            )
        elif config.optimizer == "adam":
            self.optimizer = Adam(
                model.parameters(),
                lr=config.learning_rate,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer!r}. Use 'sgd' or 'adam'.")

        if config.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif config.lr_scheduler == "step":
            self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif config.lr_scheduler is None:
            self.scheduler = None
        else:
            raise ValueError(
                f"Unsupported lr_scheduler: {config.lr_scheduler!r}. Use 'cosine', 'step', or None."
            )

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_text = PrettyPrinter().format(config)
        (output_dir / "config.yaml").write_text(config_text)

    def train_epoch(self, data_loader: DataLoader, epoch: int = 0, total_epochs: int = 0) -> float:
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        weights = self.config.exit_loss_weights
        total_loss = 0.0
        num_batches = 0

        desc = f"Epoch {epoch:>3}/{total_epochs}" if total_epochs else "Training"
        pbar = tqdm(data_loader, desc=desc, leave=False, unit="batch")

        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            exit_outputs = self.model(inputs)

            loss = sum(
                w * criterion(out.logits, targets)
                for w, out in zip(weights, exit_outputs)
            )
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / num_batches if num_batches > 0 else 0.0

    def evaluate(self, data_loader: DataLoader) -> float:
        """Return accuracy of the final head on the given data loader."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                exit_outputs = self.model(inputs)
                # final head is always the last output
                preds = exit_outputs[-1].predicted_class
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        return correct / total if total > 0 else 0.0

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int | None = None,
        eval_loader: DataLoader | None = None,
    ) -> None:
        epochs = num_epochs if num_epochs is not None else self.config.num_epochs
        total = epochs
        for epoch in range(1, total + 1):
            train_loss = self.train_epoch(train_loader, epoch=epoch, total_epochs=total)
            lr = self.optimizer.param_groups[0]["lr"]

            if eval_loader is not None:
                val_acc = self.evaluate(eval_loader)
                print(
                    f"Epoch {epoch:>3}/{total}  "
                    f"loss={train_loss:.4f}  "
                    f"val_acc={val_acc:.4f}  "
                    f"lr={lr:.6f}"
                )
            else:
                print(
                    f"Epoch {epoch:>3}/{total}  "
                    f"loss={train_loss:.4f}  "
                    f"lr={lr:.6f}"
                )

            if self.scheduler is not None:
                self.scheduler.step()
