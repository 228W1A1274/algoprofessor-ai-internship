"""
utils/metrics.py
================
Evaluation metrics for DealHunter price predictions.
"""

import numpy as np


def mae(y_true, y_pred):
    """Mean Absolute Error — average dollar error."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error — scale-invariant, main metric."""
    a, p = np.array(y_true, float), np.array(y_pred, float)
    mask = a != 0
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def rmse(y_true, y_pred):
    """Root Mean Squared Error — penalises large errors."""
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def accuracy_within(y_true, y_pred, tol=20):
    """% of predictions within ±tol% of the true price."""
    a, p = np.array(y_true, float), np.array(y_pred, float)
    return float(np.mean(np.abs(a - p) / np.maximum(a, 1e-9) * 100 <= tol) * 100)


def compute_metrics(y_true, y_pred, model_name, n_failed=0):
    """Compute all metrics with NaN guard for empty results."""
    n_total = len(y_true) + n_failed

    if len(y_true) == 0:
        print(f"  ⚠️  {model_name}: 0 successful predictions out of {n_total}")
        return {
            "model": model_name,
            "n_evaluated": 0,
            "n_failed": n_failed,
            "parse_success_rate": 0.0,
            "mae": None,
            "mape": None,
            "rmse": None,
            "accuracy_within_10pct": None,
            "accuracy_within_20pct": None,
            "accuracy_within_50pct": None,
        }

    return {
        "model":                  model_name,
        "n_evaluated":            len(y_true),
        "n_failed":               n_failed,
        "parse_success_rate":     round(len(y_true) / n_total * 100, 2),
        "mae":                    round(mae(y_true, y_pred), 2),
        "mape":                   round(mape(y_true, y_pred), 2),
        "rmse":                   round(rmse(y_true, y_pred), 2),
        "accuracy_within_10pct":  round(accuracy_within(y_true, y_pred, 10), 2),
        "accuracy_within_20pct":  round(accuracy_within(y_true, y_pred, 20), 2),
        "accuracy_within_50pct":  round(accuracy_within(y_true, y_pred, 50), 2),
    }
