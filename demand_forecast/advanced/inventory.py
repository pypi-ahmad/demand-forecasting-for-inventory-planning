"""Inventory-oriented metrics and quantile reorder signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InventoryScore:
    under_units: float  # sum of (actual - pred)+  ≈ stockout volume
    over_units: float  # sum of (pred - actual)+   ≈ excess
    asymmetric_cost: float
    service_level_hit_rate: float  # fraction of periods with pred >= actual
    pinball_loss: float


def asymmetric_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    underage_cost: float = 4.0,
    overage_cost: float = 1.0,
) -> InventoryScore:
    """Newsvendor-style period costs; underage usually > overage for retail."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    under = np.maximum(y_true - y_pred, 0.0)
    over = np.maximum(y_pred - y_true, 0.0)
    cost = float(underage_cost * under.sum() + overage_cost * over.sum())
    hit = float(np.mean(y_pred >= y_true))
    # pinball at tau = under/(under+over) newsvendor critical fractile
    tau = underage_cost / (underage_cost + overage_cost)
    pinball = float(np.mean(np.where(y_true >= y_pred, tau * (y_true - y_pred), (1 - tau) * (y_pred - y_true))))
    return InventoryScore(
        under_units=float(under.sum()),
        over_units=float(over.sum()),
        asymmetric_cost=cost,
        service_level_hit_rate=hit,
        pinball_loss=pinball,
    )


def quantile_order_quantity(
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    service_level: float = 0.9,
) -> np.ndarray:
    """Interpolate order qty between median/point and upper PI for target service.

    Rough linear interpolation in probability space:
    service 0.5 → point, service 0.9 → upper (approx q90 band).
    """
    service_level = float(np.clip(service_level, 0.5, 0.99))
    # map 0.5..0.9 → 0..1 toward upper
    w = (service_level - 0.5) / 0.4
    w = float(np.clip(w, 0.0, 1.0))
    return (1 - w) * np.asarray(point, dtype=float) + w * np.asarray(upper, dtype=float)


def safety_buffer_from_band(
    point: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Implied safety stock ≈ upper - point (non-negative)."""
    return np.maximum(np.asarray(upper, dtype=float) - np.asarray(point, dtype=float), 0.0)
