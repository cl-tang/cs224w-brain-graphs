"""Training configuration."""

from dataclasses import dataclass
from .defaults import (
    DEFAULT_SEED, DEFAULT_VAL_RATIO, DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_WEIGHT_DECAY, DEFAULT_PATIENCE,
)


@dataclass
class TrainingConfig:
    """Training hyperparameters. Uses Adam optimizer and MAE loss."""

    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY

    patience: int = DEFAULT_PATIENCE  # early stopping
    min_delta: float = 0.001

    seed: int = DEFAULT_SEED
    val_ratio: float = DEFAULT_VAL_RATIO

    def __str__(self):
        return f"TrainingConfig(epochs={self.epochs}, batch={self.batch_size}, lr={self.learning_rate})"
