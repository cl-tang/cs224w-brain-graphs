from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def plot_training_curves(
    epochs: Sequence[float],
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    train_maes: Sequence[float],
    val_maes: Sequence[float],
    output_path: Path | str,
    show: bool = True,
) -> None:
    """Plot and save training/validation loss and MAE curves."""
    output_path = Path(output_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, train_losses, label="Train")
    axes[0].plot(epochs, val_losses, label="Validation")
    axes[0].set_title("Loss (L1) over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_maes, label="Train")
    axes[1].plot(epochs, val_maes, label="Validation")
    axes[1].set_title("Mean Absolute Error over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved training curves to {output_path.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)

