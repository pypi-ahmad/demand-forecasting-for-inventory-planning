"""Holdout and backtest metrics for unit-demand forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def forecast_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_train: np.ndarray | pd.Series,
    *,
    mase_period: int = 1,
) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE, sMAPE, MASE, and mean bias.

    MASE uses seasonal naive scale on *train* with period ``mase_period``
    (falls back to 1-step naive scale if unstable).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    smape = float(
        np.mean(2.0 * np.abs(err) / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8))
        * 100.0
    )
    m = max(1, min(int(mase_period), max(1, len(y_train) // 2)))
    scale = float(np.mean(np.abs(y_train[m:] - y_train[:-m])))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.mean(np.abs(np.diff(y_train)))) + 1e-8
    mase = float(mae / scale)
    bias = float(np.mean(y_pred - y_true))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
        "MASE": mase,
        "bias": bias,
    }


def metrics_table(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build a sorted metrics table (by MASE ascending)."""
    frame = pd.DataFrame(rows).T
    cols = [c for c in ["MAE", "RMSE", "MAPE", "sMAPE", "MASE", "bias"] if c in frame.columns]
    frame = frame[cols].sort_values("MASE")
    return frame


def pi_coverage(
    y_true: np.ndarray | pd.Series,
    lower: np.ndarray | pd.Series,
    upper: np.ndarray | pd.Series,
) -> float:
    """Fraction of actuals inside [lower, upper]."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def summarize_backtest(fold_metrics: list[dict[str, Any]]) -> pd.DataFrame:
    """Average metrics across rolling-origin folds."""
    if not fold_metrics:
        return pd.DataFrame()
    frame = pd.DataFrame(fold_metrics)
    num = frame.select_dtypes(include=[np.number])
    return num.agg(["mean", "std", "min", "max"])
