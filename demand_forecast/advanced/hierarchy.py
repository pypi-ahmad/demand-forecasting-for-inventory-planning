"""Bottom-up hierarchical forecasting and reconciliation to totals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass
class HierarchyResult:
    bottom_forecasts: dict[str, np.ndarray]
    total_bottom_up: np.ndarray
    series_used: list[str]
    notes: list[str]


def _pick_period(n: int) -> int:
    for p in (52, 26, 13, 8, 4):
        if n >= 2 * p + 2:
            return p
    return max(2, n // 4)


def forecast_bottom_series(
    series: pd.Series,
    h: int,
    *,
    seasonal: str = "mul",
) -> np.ndarray:
    """HW forecast for one bottom-level non-negative series (explosion-safe)."""
    y = series.astype(float).clip(lower=0.0)
    if (y <= 0).all():
        return np.zeros(h)
    hist = y.values
    hist_max = float(np.max(hist))
    hist_mean = float(np.mean(hist[-min(8, len(hist)) :]))
    # Multiplicative HW is unstable on sparse/zero-inflated bottoms → force additive
    zero_share = float(np.mean(hist == 0))
    use_seasonal = "add" if (zero_share > 0.2 or seasonal != "mul") else "mul"
    period = _pick_period(len(y))
    y_fit = y.clip(lower=1e-3) if use_seasonal == "mul" else y
    # ensure regular weekly index for statsmodels
    if not isinstance(y_fit.index, pd.DatetimeIndex):
        y_fit = y_fit.copy()
    try:
        y_fit = y_fit.asfreq("W-SUN", fill_value=float(y_fit.min() if use_seasonal == "mul" else 0.0))
    except Exception:
        pass
    try:
        if len(y_fit) >= 2 * period + 2:
            model = ExponentialSmoothing(
                y_fit,
                trend="add",
                seasonal=use_seasonal,
                seasonal_periods=period,
                initialization_method="estimated",
            ).fit(optimized=True)
            point = np.asarray(model.forecast(h), dtype=float)
            # hard cap: no bottom can exceed 3× historical max (stops mul explosions)
            cap = max(hist_max * 3.0, hist_mean * 5.0, 1.0)
            return np.clip(point, 0, cap)
    except Exception:
        pass
    sp = min(period, len(hist))
    if len(hist) >= sp and sp > 0:
        return np.array([hist[-sp + (i % sp)] for i in range(h)], dtype=float)
    return np.full(h, float(hist[-1] if len(hist) else 0.0))


def build_bottom_panel(
    transactions: pd.DataFrame,
    *,
    date_col: str,
    qty_col: str,
    group_cols: list[str],
    freq: str = "W-SUN",
    min_total_qty: float = 50.0,
) -> pd.DataFrame:
    """Wide panel of weekly unit demand by group key.

    Columns are string keys ``a|b`` for multi-column groups.
    Only groups with total quantity >= min_total_qty are kept (stability).
    """
    df = transactions.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce")
    df = df.dropna(subset=[date_col, qty_col])
    df = df[df[qty_col] > 0]
    for c in group_cols:
        df[c] = df[c].astype(str).fillna("UNK")

    df["_key"] = df[group_cols].agg("|".join, axis=1)
    totals = df.groupby("_key")[qty_col].sum()
    keep = totals[totals >= min_total_qty].index
    df = df[df["_key"].isin(keep)]

    # Reliable weekly panel: Grouper on dates × key (avoids multiindex resample pitfalls)
    g = (
        df.groupby([pd.Grouper(key=date_col, freq=freq), "_key"], observed=True)[qty_col]
        .sum()
        .unstack(level=1)
        .fillna(0.0)
        .sort_index()
    )
    return g


def bottom_up_forecast(
    panel: pd.DataFrame,
    h: int,
    *,
    seasonal: str = "mul",
    top_k: int | None = None,
) -> HierarchyResult:
    """Forecast each bottom series and sum (bottom-up reconciliation).

    Parameters
    ----------
    top_k:
        If set, only the top_k series by train sum are modeled individually;
        the residual 'other' bucket is also forecast and added.
    """
    notes: list[str] = []
    if panel.empty:
        return HierarchyResult({}, np.zeros(h), [], ["empty panel"])

    col_sums = panel.sum(axis=0).sort_values(ascending=False)
    if top_k is not None and top_k < len(col_sums):
        main_cols = list(col_sums.index[:top_k])
        other = panel.drop(columns=main_cols).sum(axis=1)
        other.name = "__OTHER__"
        work = panel[main_cols].copy()
        work["__OTHER__"] = other
        notes.append(f"top_k={top_k} bottoms + OTHER residual bucket")
    else:
        work = panel
        notes.append(f"all {work.shape[1]} bottom series")

    forecasts: dict[str, np.ndarray] = {}
    for col in work.columns:
        s = work[col].astype(float)
        # drop leading zeros for fitting stability but keep length for index align
        forecasts[str(col)] = forecast_bottom_series(s, h, seasonal=seasonal)

    total = np.sum(np.vstack(list(forecasts.values())), axis=0)
    notes.append("reconciliation=bottom_up_sum")
    return HierarchyResult(
        bottom_forecasts=forecasts,
        total_bottom_up=np.clip(total, 0, None),
        series_used=list(forecasts.keys()),
        notes=notes,
    )
