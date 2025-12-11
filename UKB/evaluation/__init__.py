"""Evaluation utilities for brain age prediction."""

from .metrics import (
    compute_regression_metrics, compute_brain_age_gap,
    summarize_results, compare_models,
)
from .plots import plot_training_curves, plot_predictions_scatter

__all__ = [
    "compute_regression_metrics", "compute_brain_age_gap",
    "summarize_results", "compare_models",
    "plot_training_curves", "plot_predictions_scatter",
]
