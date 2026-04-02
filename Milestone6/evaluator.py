"""
evaluator.py
────────────
Evaluation metric functions for LLM price prediction benchmarking.

Metrics:
  - MAE  : Mean Absolute Error          — average dollar error
  - MAPE : Mean Absolute Percentage Error — scale-independent (primary metric)
  - RMSE : Root Mean Squared Error      — penalizes large errors

All functions handle None values in y_pred by filtering them out and
reporting a 'coverage' score (what % of predictions were parseable).

Usage:
    from evaluator import compute_metrics, build_leaderboard

    metrics = compute_metrics(y_true_list, y_pred_list)
    # → {"MAE": 32.5, "MAPE": 6.2, "RMSE": 48.1, "coverage": 95.0, ...}

    leaderboard_df = build_leaderboard(model_results_dict, y_true_list)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ── Core Metric Function ──────────────────────────────────────────

def compute_metrics(y_true: list, y_pred: list) -> dict:
    """
    Compute MAE, MAPE, RMSE, and prediction coverage.

    Pairs where y_pred is None (parse failure) are excluded from metric
    computation but counted against the coverage score.

    Args:
        y_true (list[float]): Ground truth prices.
        y_pred (list[float | None]): Predicted prices; may contain None.

    Returns:
        dict with keys:
            "MAE"      (float | None): Mean Absolute Error in dollars.
            "MAPE"     (float | None): Mean Absolute Percentage Error in %.
            "RMSE"     (float | None): Root Mean Squared Error in dollars.
            "coverage" (float)       : % of products with a valid prediction.
            "n_valid"  (int)         : Number of valid (non-None) predictions.
            "n_total"  (int)         : Total number of products evaluated.
    """
    n_total = len(y_true)

    # Filter out None predictions
    valid_pairs = [
        (true, pred)
        for true, pred in zip(y_true, y_pred)
        if pred is not None
    ]

    n_valid   = len(valid_pairs)
    coverage  = round((n_valid / n_total) * 100, 1) if n_total > 0 else 0.0

    if n_valid == 0:
        return {
            "MAE": None, "MAPE": None, "RMSE": None,
            "coverage": coverage, "n_valid": 0, "n_total": n_total
        }

    yt = np.array([p[0] for p in valid_pairs], dtype=float)
    yp = np.array([p[1] for p in valid_pairs], dtype=float)

    # MAE ─ mean(|true - pred|)
    mae = float(mean_absolute_error(yt, yp))

    # MAPE ─ mean(|true - pred| / true) × 100  [skip zero-valued truths]
    nonzero = yt != 0
    mape = float(
        np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero])) * 100
    ) if nonzero.any() else None

    # RMSE ─ sqrt(mean((true - pred)²))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))

    return {
        "MAE":      round(mae,  2),
        "MAPE":     round(mape, 2) if mape is not None else None,
        "RMSE":     round(rmse, 2),
        "coverage": coverage,
        "n_valid":  n_valid,
        "n_total":  n_total
    }


# ── Leaderboard Builder ───────────────────────────────────────────

def build_leaderboard(model_results: dict, y_true: list) -> pd.DataFrame:
    """
    Build a ranked leaderboard DataFrame from multiple model predictions.

    Primary sort: MAPE ascending (lower = better, more accurate model).
    Secondary sort: Coverage descending (higher = fewer parse failures).

    Args:
        model_results (dict): {
            "Model Display Name": [pred1, pred2, ...],
            ...
        }
        y_true (list[float]): Ground truth prices (same order as predictions).

    Returns:
        pd.DataFrame: Leaderboard with columns:
            Rank, Model, MAE ($), MAPE (%), RMSE ($), Coverage, Valid/Total
    """
    rows = []

    for model_name, y_pred in model_results.items():
        m = compute_metrics(y_true, y_pred)
        rows.append({
            "Model":        model_name,
            "MAE ($)":      f"${m['MAE']:.2f}"   if m["MAE"]  is not None else "N/A",
            "MAPE (%)":     f"{m['MAPE']:.2f}%"  if m["MAPE"] is not None else "N/A",
            "RMSE ($)":     f"${m['RMSE']:.2f}"  if m["RMSE"] is not None else "N/A",
            "Coverage":     f"{m['coverage']}%",
            "Valid/Total":  f"{m['n_valid']}/{m['n_total']}",
            # raw numeric values for sorting — dropped before display
            "_mape_sort":   m["MAPE"]     if m["MAPE"]     is not None else float("inf"),
            "_cov_sort":    m["coverage"] if m["coverage"] is not None else 0.0,
        })

    # Sort: best MAPE first, then best coverage as tiebreaker
    rows.sort(key=lambda r: (r["_mape_sort"], -r["_cov_sort"]))

    df = pd.DataFrame(rows)
    df.insert(0, "Rank", [f"#{i+1}" for i in range(len(df))])
    df = df.drop(columns=["_mape_sort", "_cov_sort"])

    return df.reset_index(drop=True)


# ── Per-Product Error Breakdown ───────────────────────────────────

def per_product_errors(
    names:    list,
    y_true:   list,
    y_pred:   list,
    model_name: str = "Model"
) -> pd.DataFrame:
    """
    Compute per-product absolute error and percentage error.

    Useful for spotting which product categories a model struggles with.

    Args:
        names      (list[str]):         Product names.
        y_true     (list[float]):       Ground truth prices.
        y_pred     (list[float|None]):  Predicted prices.
        model_name (str):               Label for the prediction column.

    Returns:
        pd.DataFrame with columns:
            Product, Ground Truth ($), Prediction ($), Abs Error ($), % Error
    """
    rows = []
    for name, true, pred in zip(names, y_true, y_pred):
        if pred is not None:
            abs_err = abs(true - pred)
            pct_err = (abs_err / true * 100) if true != 0 else None
            rows.append({
                "Product":          name,
                "Ground Truth ($)": f"${true:.2f}",
                f"{model_name} ($)": f"${pred:.2f}",
                "Abs Error ($)":    f"${abs_err:.2f}",
                "% Error":          f"{pct_err:.1f}%" if pct_err is not None else "N/A"
            })
        else:
            rows.append({
                "Product":          name,
                "Ground Truth ($)": f"${true:.2f}",
                f"{model_name} ($)": "FAILED",
                "Abs Error ($)":    "—",
                "% Error":          "—"
            })

    return pd.DataFrame(rows)
