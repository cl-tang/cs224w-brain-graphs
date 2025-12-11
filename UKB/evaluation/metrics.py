"""Evaluation metrics for brain age prediction."""

import numpy as np
from scipy import stats


def compute_regression_metrics(y_true, y_pred):
    """Compute MAE, RMSE, R², and correlations."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)

    return {
        "mae": mae, "rmse": rmse, "mse": mse, "r2": r2,
        "pearson_r": pearson_r, "pearson_p": pearson_p,
        "spearman_r": spearman_r, "spearman_p": spearman_p,
    }


def compute_brain_age_gap(y_true, y_pred):
    """Brain age gap = predicted - chronological age."""
    return np.asarray(y_pred).flatten() - np.asarray(y_true).flatten()


def summarize_results(y_true, y_pred, dataset_name="Test"):
    """Generate formatted results summary."""
    metrics = compute_regression_metrics(y_true, y_pred)
    return f"""
{'='*50}
{dataset_name} Results
{'='*50}
  N samples:     {len(y_true)}
  MAE:           {metrics['mae']:.3f} years
  RMSE:          {metrics['rmse']:.3f} years
  R²:            {metrics['r2']:.4f}
  Pearson r:     {metrics['pearson_r']:.4f}
  Spearman ρ:    {metrics['spearman_r']:.4f}
{'='*50}"""


def compare_models(results):
    """Generate comparison table for multiple models."""
    lines = [
        "\n" + "=" * 70,
        "Model Comparison",
        "=" * 70,
        f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'r':>8}",
        "-" * 70,
    ]
    for name, metrics in results.items():
        lines.append(
            f"{name:<20} {metrics['mae']:>8.3f} {metrics['rmse']:>8.3f} "
            f"{metrics['r2']:>8.4f} {metrics['pearson_r']:>8.4f}"
        )
    lines.append("=" * 70)
    return "\n".join(lines)
