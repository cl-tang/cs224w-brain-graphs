"""Plotting utilities for brain age prediction."""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_training_curves(history, save_path=None, show=False, title="Training Progress"):
    """Plot training and validation curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    has_pearson_r = "val_pearson_r" in history and history["val_pearson_r"]

    fig, axes = plt.subplots(1, 3 if has_pearson_r else 2, figsize=(15 if has_pearson_r else 12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
    axes[1].plot(epochs, history["train_mae"], label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_mae"], label="Val", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (years)")
    axes[1].set_title("Mean Absolute Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Pearson r
    if has_pearson_r:
        axes[2].plot(epochs, history["val_pearson_r"], label="Val r", linewidth=2, color='green')
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Pearson r")
        axes[2].set_title("Validation Correlation")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        best_idx = np.argmax(history["val_pearson_r"])
        axes[2].axhline(y=history["val_pearson_r"][best_idx], color='gray', linestyle='--', alpha=0.5)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_predictions_scatter(y_true, y_pred, save_path=None, show=False,
                             title="Predicted vs True Age", metrics=None):
    """Plot predicted vs true age scatter."""
    from scipy import stats
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if metrics is None:
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'pearson_r': stats.pearsonr(y_true, y_pred)[0],
            'spearman_r': stats.spearmanr(y_true, y_pred)[0],
        }

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.5, s=20, c='steelblue')

    # Identity line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Identity')

    # Regression line
    slope, intercept, _, _, _ = stats.linregress(y_true, y_pred)
    ax.plot([min_val, max_val], [slope*min_val + intercept, slope*max_val + intercept],
            'g-', lw=2, label=f'Fit (R²={metrics.get("r2", 0):.3f})')

    ax.set_xlabel("Chronological Age (years)")
    ax.set_ylabel("Predicted Brain Age (years)")
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    text = f"MAE: {metrics.get('mae', 0):.2f}\nRMSE: {metrics.get('rmse', 0):.2f}\n" \
           f"R²: {metrics.get('r2', 0):.4f}\nr: {metrics.get('pearson_r', 0):.4f}"
    ax.text(0.97, 0.03, text, transform=ax.transAxes, fontsize=10,
            va='bottom', ha='right', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    ax.set_aspect('equal')

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
