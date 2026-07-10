"""Explosion-safe bottom-up hierarchical weekly forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass
class BottomUpResult:
    name: str
    point: np.ndarray
    group_keys: list[str]
    details: str


def _hw(s: pd.Series, h: int, period: int, seasonal: str) -> np.ndarray:
    y = s.astype(float)
    if seasonal == "mul":
        y = y.clip(lower=1e-3)
    if len(y) < 2 * period + 2:
        sp = max(1, min(period, len(y)))
        hist = y.values
        return np.array([hist[-sp + (i % sp)] for i in range(h)], dtype=float)
    try:
        y = y.asfreq("W-SUN", method=None)
        y = y.fillna(y.median() if seasonal == "mul" else 0.0)
    except Exception:
        pass
    model = ExponentialSmoothing(
        y,
        trend="add",
        seasonal=seasonal,
        seasonal_periods=period,
        initialization_method="estimated",
    ).fit(optimized=True)
    return np.asarray(model.forecast(h), dtype=float)


def bottom_up_total(
    transactions: pd.DataFrame,
    *,
    date_col: str,
    qty_col: str,
    group_cols: list[str],
    train_index: pd.DatetimeIndex,
    h: int,
    period: int = 52,
    min_total_qty: float = 50.0,
) -> BottomUpResult:
    """Forecast each group on the train window and sum to a total forecast."""
    df = transactions.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce")
    df = df.dropna(subset=[date_col, qty_col])
    df = df[df[qty_col] > 0]
    # only history available at train end
    train_end = train_index.max()
    df = df[df[date_col] <= train_end]

    for c in group_cols:
        df[c] = df[c].astype(str).fillna("UNK")
    df["_key"] = df[group_cols].agg("|".join, axis=1)

    totals = df.groupby("_key")[qty_col].sum()
    keys = totals[totals >= min_total_qty].index.tolist()
    preds: list[np.ndarray] = []
    used: list[str] = []

    for key in keys:
        g = df[df["_key"] == key]
        s = (
            g.set_index(date_col)[qty_col]
            .resample("W-SUN")
            .sum()
            .reindex(train_index)
            .fillna(0.0)
            .astype(float)
        )
        zero_share = float((s == 0).mean())
        seas = "add" if zero_share > 0.15 else "mul"
        p = period
        if len(s) < 2 * p + 2:
            for cand in (26, 13, 8, 4):
                if len(s) >= 2 * cand + 2:
                    p = cand
                    break
        try:
            fc = _hw(s, h, p, seas)
        except Exception:
            sp = max(1, min(p, len(s)))
            hist = s.values
            fc = np.array([hist[-sp + (i % sp)] for i in range(h)], dtype=float)
        cap = max(float(s.max()) * 3.0, float(s.tail(min(8, len(s))).mean()) * 5.0, 1.0)
        preds.append(np.clip(fc, 0.0, cap))
        used.append(key)

    if not preds:
        point = np.zeros(h)
    else:
        point = np.sum(np.vstack(preds), axis=0)

    return BottomUpResult(
        name="bu_" + "_".join(group_cols).lower(),
        point=np.clip(point, 0, None),
        group_keys=used,
        details=f"bottom-up groups={group_cols} n={len(used)} period≈{period}",
    )


def volume_scale(point: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Scale forecast so horizon sum matches reference horizon sum."""
    p = np.asarray(point, dtype=float)
    r = np.asarray(reference, dtype=float)
    if p.sum() <= 0:
        return p
    return p * (r.sum() / p.sum())


def blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """alpha * a + (1-alpha) * b."""
    return alpha * np.asarray(a, dtype=float) + (1.0 - alpha) * np.asarray(b, dtype=float)
