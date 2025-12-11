"""Training loop for brain age models."""

from pathlib import Path
from typing import Dict
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from scipy import stats

from ..configs import TrainingConfig


class EarlyStopping:
    """Early stopping based on validation loss."""

    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_value is None:
            self.best_value = val_loss
        elif val_loss > self.best_value - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_value = val_loss
            self.counter = 0
        return self.should_stop


class Trainer:
    """Trainer for brain age models. Uses Adam optimizer and MAE loss."""

    def __init__(self, model, train_loader, val_loader, config, device=None, checkpoint_dir=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

        # Fixed: Adam optimizer, MAE loss
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.loss_fn = nn.L1Loss()  # MAE
        self.early_stopping = EarlyStopping(config.patience, config.min_delta)

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [], "val_loss": [], "train_mae": [],
            "val_mae": [], "val_pearson_r": [], "lr": [],
        }
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def _run_epoch(self, loader, training=True):
        """Run one epoch, return (loss, mae)."""
        self.model.train() if training else self.model.eval()
        total_loss, total_mae, total_samples = 0.0, 0.0, 0

        with torch.set_grad_enabled(training):
            for batch in loader:
                batch = batch.to(self.device, non_blocking=True)
                pred = self.model(batch)
                loss = self.loss_fn(pred, batch.y)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                mae = torch.mean(torch.abs(pred.detach() - batch.y.detach()))
                batch_size = batch.num_graphs
                total_loss += loss.item() * batch_size
                total_mae += mae.item() * batch_size
                total_samples += batch_size

        return total_loss / max(1, total_samples), total_mae / max(1, total_samples)

    def _run_validation(self, loader):
        """Run validation, return (loss, mae, pearson_r)."""
        self.model.eval()
        all_preds, all_targets = [], []
        total_loss, total_samples = 0.0, 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device, non_blocking=True)
                pred = self.model(batch)
                loss = self.loss_fn(pred, batch.y)

                total_loss += loss.item() * batch.num_graphs
                total_samples += batch.num_graphs
                all_preds.append(pred.cpu())
                all_targets.append(batch.y.cpu())

        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        avg_loss = total_loss / max(1, total_samples)
        avg_mae = np.mean(np.abs(preds - targets))
        pearson_r, _ = stats.pearsonr(targets.flatten(), preds.flatten())

        return avg_loss, avg_mae, pearson_r

    def train(self, verbose=True):
        """Train the model."""
        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_mae = self._run_epoch(self.train_loader, training=True)
            val_loss, val_mae, val_pearson_r = self._run_validation(self.val_loader)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_mae"].append(train_mae)
            self.history["val_mae"].append(val_mae)
            self.history["val_pearson_r"].append(val_pearson_r)
            self.history["lr"].append(current_lr)

            # Save best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                if self.checkpoint_dir:
                    self.save_checkpoint(self.checkpoint_dir / "best_model.pt")

            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"Epoch {epoch:3d}/{self.config.epochs} | "
                      f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | MAE: {val_mae:.3f}")

            if self.early_stopping(val_loss):
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        if verbose:
            print(f"\nDone in {time.time() - start_time:.1f}s, best val loss: {self.best_val_loss:.4f}")

        return self.history

    def evaluate(self, loader):
        """Evaluate model on a loader."""
        self.model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device, non_blocking=True)
                pred = self.model(batch)
                all_preds.append(pred.cpu())
                all_targets.append(batch.y.cpu())

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        mae = torch.mean(torch.abs(preds - targets)).item()
        mse = torch.mean((preds - targets) ** 2).item()

        return {
            "mae": mae, "mse": mse, "rmse": mse ** 0.5,
            "predictions": preds.numpy(), "targets": targets.numpy(),
        }

    def save_checkpoint(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": len(self.history["train_loss"]),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))


def set_seed(seed):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
