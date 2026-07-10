"""Classical / statistical demand forecasters (native libraries)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    from pmdarima import auto_arima
except ImportError:  # pragma: no cover
    auto_arima = None  # type: ignore[assignment]


@dataclass
class ClassicalForecast:
    name: str
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    residuals: pd.Series | None
    fitted_model: Any
    details: str


def seasonal_periods_for_series(n: int, grain: str) -> list[int]:
    """Candidate seasonal periods that fit at least 2 full cycles in *n* points."""
    if grain == "weekly":
        candidates = [4, 8, 13, 26, 52]
    else:
        candidates = [7, 14, 28]
    return [p for p in candidates if n >= 2 * p + 2]


def mase_period_for_series(n: int, grain: str) -> int:
    """Prefer annual cycle when enough history exists."""
    if grain == "weekly":
        return 52 if n > 60 else 4
    return 7 if n > 21 else 1


def _residual_sigma(resid: pd.Series | np.ndarray | None) -> float:
    if resid is None:
        return 1.0
    r = pd.Series(resid).dropna()
    if len(r) < 3:
        return 1.0
    s = float(r.std(ddof=1))
    return s if np.isfinite(s) and s > 0 else 1.0


def _pi(point: np.ndarray, sigma: float, z: float = 1.2815515655446004) -> tuple[np.ndarray, np.ndarray]:
    """Approx central 80% PI from residual sigma (z≈0.9 quantile)."""
    return point - z * sigma, point + z * sigma


def forecast_seasonal_naive(
    train: pd.Series, h: int, period: int
) -> ClassicalForecast:
    hist = train.astype(float).values
    sp = max(1, min(period, len(hist)))
    if len(hist) < sp:
        point = np.full(h, float(hist[-1]))
    else:
        point = np.array([hist[-sp + (i % sp)] for i in range(h)], dtype=float)
    if len(hist) > sp:
        resid = pd.Series(hist[sp:] - hist[:-sp])
    else:
        resid = train.diff().dropna()
    sigma = _residual_sigma(resid)
    lo, hi = _pi(point, sigma)
    return ClassicalForecast(
        name=f"seasonal_naive_m{sp}",
        point=np.clip(point, 0, None),
        lower=lo,
        upper=hi,
        residuals=resid,
        fitted_model=None,
        details=f"Seasonal naive, period={sp}",
    )


def forecast_holt_winters(
    train: pd.Series,
    h: int,
    *,
    seasonal: str,
    period: int,
) -> ClassicalForecast:
    """Holt-Winters / ETS-style exponential smoothing via statsmodels."""
    y = train.astype(float).copy()
    if seasonal == "mul":
        y = y.clip(lower=1e-3)
    model = ExponentialSmoothing(
        y,
        trend="add",
        seasonal=seasonal,
        seasonal_periods=period,
        initialization_method="estimated",
    ).fit(optimized=True)
    point = np.asarray(model.forecast(h), dtype=float)
    resid = (y - model.fittedvalues).dropna()
    sigma = _residual_sigma(resid)
    lo, hi = _pi(point, sigma)
    return ClassicalForecast(
        name=f"holt_winters_{seasonal}_m{period}",
        point=np.clip(point, 0, None),
        lower=lo,
        upper=hi,
        residuals=resid,
        fitted_model=model,
        details=f"ExponentialSmoothing trend=add seasonal={seasonal} period={period}",
    )


def forecast_auto_arima(
    train: pd.Series,
    h: int,
    *,
    seasonal_period: int,
    seed: int = 42,
) -> ClassicalForecast:
    if auto_arima is None:
        raise ImportError("pmdarima is required for auto_arima")
    y = train.astype(float)
    sp = seasonal_period if len(y) >= 2 * seasonal_period else 1
    seasonal = sp > 1
    model = auto_arima(
        y,
        seasonal=seasonal,
        m=sp if seasonal else 1,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=3,
        max_q=3,
        max_P=1,
        max_Q=1,
        max_d=2,
        max_D=1,
        n_jobs=1,
        random_state=seed,
    )
    fc, conf = model.predict(n_periods=h, return_conf_int=True, alpha=0.2)
    point = np.asarray(fc, dtype=float)
    lower = np.asarray(conf[:, 0], dtype=float)
    upper = np.asarray(conf[:, 1], dtype=float)
    resid = pd.Series(model.resid(), index=y.index[-len(model.resid()) :])
    return ClassicalForecast(
        name=f"auto_arima_m{sp}",
        point=np.clip(point, 0, None),
        lower=lower,
        upper=upper,
        residuals=resid,
        fitted_model=model,
        details=f"pmdarima order={model.order} seasonal_order={model.seasonal_order}",
    )


def build_classical_candidates(
    train: pd.Series,
    h: int,
    grain: str,
    *,
    seed: int = 42,
) -> list[ClassicalForecast]:
    """Fit a production shortlist of classical models that fit the series length."""
    n = len(train)
    periods = seasonal_periods_for_series(n, grain)
    out: list[ClassicalForecast] = []

    # Always include seasonal naive at the longest feasible period + short cycle
    naive_periods = sorted({p for p in periods + ([52] if grain == "weekly" and n > 52 else [])})
    if not naive_periods:
        naive_periods = [1]
    for sp in {naive_periods[-1], naive_periods[0]}:
        try:
            out.append(forecast_seasonal_naive(train, h, sp))
        except Exception:
            continue

    for period in periods:
        for seasonal in ("add", "mul"):
            try:
                out.append(
                    forecast_holt_winters(train, h, seasonal=seasonal, period=period)
                )
            except Exception:
                continue

    # One auto_arima with a mid-range seasonal period (fast bounds)
    arima_m = 4 if grain == "weekly" else 7
    if arima_m not in periods and periods:
        arima_m = periods[0]
    try:
        out.append(forecast_auto_arima(train, h, seasonal_period=arima_m, seed=seed))
    except Exception:
        pass

    # de-dupe by name keeping first
    seen: set[str] = set()
    uniq: list[ClassicalForecast] = []
    for c in out:
        if c.name in seen:
            continue
        seen.add(c.name)
        uniq.append(c)
    return uniq
