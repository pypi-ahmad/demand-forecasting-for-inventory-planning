"""Calendar, holiday, and promo-style features for weekly demand series."""

from __future__ import annotations

from typing import Iterable

import holidays as holidays_lib
import numpy as np
import pandas as pd


def week_index(start: pd.Timestamp, end: pd.Timestamp, freq: str = "W-SUN") -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq=freq)


def holiday_count_in_week(
    week_ends: Iterable[pd.Timestamp],
    *,
    country: str = "US",
) -> pd.Series:
    """Count public holidays falling in each week ending on ``week_ends``."""
    week_ends = pd.DatetimeIndex(week_ends)
    years = range(int(week_ends.min().year) - 1, int(week_ends.max().year) + 2)
    if country.upper() == "US":
        cal = holidays_lib.country_holidays("US", years=years)
    elif country.upper() in {"UK", "GB"}:
        cal = holidays_lib.country_holidays("GB", years=years)
    else:
        cal = holidays_lib.country_holidays(country, years=years)

    counts = []
    for end in week_ends:
        start = end - pd.Timedelta(days=6)
        n = sum(1 for d in pd.date_range(start, end, freq="D") if d.date() in cal)
        counts.append(float(n))
    return pd.Series(counts, index=week_ends, name=f"holidays_{country.lower()}")


def build_calendar_frame(
    index: pd.DatetimeIndex,
    *,
    country: str = "US",
    include_fourier: bool = True,
) -> pd.DataFrame:
    """Build exogenous feature matrix aligned to a weekly index.

    Features
    --------
    - weekofyear, month (cyclical sin/cos)
    - is_q4, is_december (retail peak flags)
    - holiday count in week
    - optional annual Fourier terms (K=2)
    """
    idx = pd.DatetimeIndex(index)
    df = pd.DataFrame(index=idx)
    woy = idx.isocalendar().week.astype(float).to_numpy()
    month = idx.month.astype(float)
    df["weekofyear"] = woy
    df["month"] = month
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df["is_q4"] = (month >= 10).astype(float)
    df["is_december"] = (month == 12).astype(float)
    df["is_january"] = (month == 1).astype(float)
    df["holiday_count"] = holiday_count_in_week(idx, country=country).to_numpy()

    if include_fourier:
        t = np.arange(len(idx), dtype=float)
        for k in (1, 2):
            df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * t / 52.0)
            df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * t / 52.0)
    return df


def aggregate_promo_proxy(
    transactions: pd.DataFrame,
    *,
    date_col: str,
    value_col: str,
    freq: str = "W-SUN",
) -> pd.Series:
    """Weekly mean discount or similar promo intensity (Superstore Discount)."""
    tmp = transactions.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, value_col])
    s = tmp.set_index(date_col)[value_col].resample(freq).mean().fillna(0.0)
    s.name = "promo_proxy"
    return s


def make_future_exog(
    history_exog: pd.DataFrame,
    h: int,
    *,
    country: str = "US",
) -> pd.DataFrame:
    """Extend calendar features h steps beyond history (promo carried as 0/ffill)."""
    if not isinstance(history_exog.index, pd.DatetimeIndex):
        raise TypeError("history_exog must have DatetimeIndex")
    last = history_exog.index[-1]
    freq = pd.infer_freq(history_exog.index) or "W-SUN"
    future_idx = pd.date_range(last, periods=h + 1, freq=freq)[1:]
    cal = build_calendar_frame(future_idx, country=country)
    # promo unknown in future → 0 (no assumed campaign)
    if "promo_proxy" in history_exog.columns:
        cal["promo_proxy"] = 0.0
    # align columns
    for c in history_exog.columns:
        if c not in cal.columns:
            cal[c] = 0.0
    return cal[history_exog.columns]
