"""Rolling-origin evaluation and reporting helpers."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from demand_forecast.advanced.inventory import asymmetric_cost
from demand_forecast.metrics import forecast_metrics


def rolling_origin_evaluate(
    y: pd.Series,
    forecast_fn: Callable[[pd.Series, int], np.ndarray],
    *,
    h: int = 8,
    n_origins: int = 8,
    min_train: int = 40,
    mase_period: int = 52,
    underage_cost: float = 4.0,
    overage_cost: float = 1.0,
) -> pd.DataFrame:
    """Expanding-window rolling origin: forecast_fn(train, h) -> point array."""
    y = y.astype(float).sort_index()
    rows = []
    max_origins = max(1, (len(y) - min_train) // max(h, 1))
    n_origins = min(n_origins, max_origins)
    for origin in range(n_origins):
        end = len(y) - origin * h
        if end - h < min_train:
            break
        y_tr = y.iloc[: end - h]
        y_te = y.iloc[end - h : end]
        try:
            point = np.asarray(forecast_fn(y_tr, h), dtype=float)
            if len(point) != h:
                point = point[:h]
            m = forecast_metrics(y_te.values, point, y_tr.values, mase_period=mase_period)
            inv = asymmetric_cost(
                y_te.values, point, underage_cost=underage_cost, overage_cost=overage_cost
            )
            rows.append(
                {
                    "origin": origin,
                    "test_end": str(y_te.index[-1].date()),
                    **m,
                    "under_units": inv.under_units,
                    "over_units": inv.over_units,
                    "asymmetric_cost": inv.asymmetric_cost,
                    "service_hit_rate": inv.service_level_hit_rate,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"origin": origin, "error": str(exc)})
    return pd.DataFrame(rows)


def summarize_rolling(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    num = df.select_dtypes(include=[np.number])
    return num.agg(["mean", "std", "median"])
