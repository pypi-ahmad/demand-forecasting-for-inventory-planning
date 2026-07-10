"""Univariate and exogenous models for advanced forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from demand_forecast.advanced.features import make_future_exog


@dataclass
class AdvForecast:
    name: str
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    details: str
    quantiles: dict[str, np.ndarray] | None = None
    extras: dict[str, Any] | None = None


def _pi(point: np.ndarray, sigma: float, z: float = 1.28155) -> tuple[np.ndarray, np.ndarray]:
    return point - z * sigma, point + z * sigma


def _sigma_from_resid(resid: np.ndarray) -> float:
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return 1.0
    s = float(np.std(r, ddof=1))
    return s if s > 0 else 1.0


def forecast_hw_log(
    y: pd.Series,
    h: int,
    *,
    period: int,
    seasonal: str = "add",
) -> AdvForecast:
    """log1p transform + HW, expm1 back-transform (variance stabilization)."""
    y = y.astype(float).clip(lower=0)
    z = np.log1p(y)
    seas = seasonal if seasonal == "add" else "add"  # log space → additive seasonal
    model = ExponentialSmoothing(
        z,
        trend="add",
        seasonal=seas if len(z) >= 2 * period + 2 else None,
        seasonal_periods=period if len(z) >= 2 * period + 2 else None,
        initialization_method="estimated",
    ).fit(optimized=True)
    zhat = np.asarray(model.forecast(h), dtype=float)
    point = np.clip(np.expm1(zhat), 0, None)
    resid = np.expm1(z.values) - np.expm1(np.asarray(model.fittedvalues, dtype=float))
    sigma = _sigma_from_resid(resid)
    lo, hi = _pi(point, sigma)
    return AdvForecast(
        name=f"hw_log1p_add_m{period}",
        point=point,
        lower=np.clip(lo, 0, None),
        upper=hi,
        details=f"log1p + ExponentialSmoothing additive seasonal m={period}",
    )


def forecast_hw_mul(
    y: pd.Series,
    h: int,
    *,
    period: int,
) -> AdvForecast:
    y = y.astype(float).clip(lower=1e-3)
    model = ExponentialSmoothing(
        y,
        trend="add",
        seasonal="mul",
        seasonal_periods=period,
        initialization_method="estimated",
    ).fit(optimized=True)
    point = np.clip(np.asarray(model.forecast(h), dtype=float), 0, None)
    resid = y.values - np.asarray(model.fittedvalues, dtype=float)
    sigma = _sigma_from_resid(resid)
    lo, hi = _pi(point, sigma)
    return AdvForecast(
        name=f"hw_mul_m{period}",
        point=point,
        lower=np.clip(lo, 0, None),
        upper=hi,
        details=f"Holt-Winters multiplicative m={period}",
    )


def forecast_sarimax_exog(
    y: pd.Series,
    exog: pd.DataFrame,
    h: int,
    *,
    order: tuple[int, int, int] = (1, 0, 1),
    seasonal_order: tuple[int, int, int, int] = (0, 1, 1, 4),
    country: str = "US",
) -> AdvForecast:
    """SARIMAX with calendar/promo exogenous regressors."""
    y = y.astype(float)
    X = exog.reindex(y.index).fillna(0.0)
    # drop constant cols
    nunique = X.nunique()
    X = X.loc[:, nunique > 1]
    if X.shape[1] == 0:
        raise ValueError("No usable exogenous columns")

    model = SARIMAX(
        y,
        exog=X,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    Xf = make_future_exog(X, h, country=country)
    fc = res.get_forecast(steps=h, exog=Xf)
    point = np.clip(np.asarray(fc.predicted_mean, dtype=float), 0, None)
    conf = fc.conf_int(alpha=0.2)
    lower = np.asarray(conf.iloc[:, 0], dtype=float)
    upper = np.asarray(conf.iloc[:, 1], dtype=float)
    return AdvForecast(
        name="sarimax_exog",
        point=point,
        lower=lower,
        upper=upper,
        details=f"SARIMAX{order}x{seasonal_order} with {list(X.columns)}",
        extras={"exog_cols": list(X.columns)},
    )


def forecast_ml_lags(
    y: pd.Series,
    exog: pd.DataFrame | None,
    h: int,
    *,
    lags: tuple[int, ...] = (1, 2, 3, 4, 13, 52),
    seed: int = 42,
) -> AdvForecast:
    """HistGradientBoosting on lags + optional exog (direct multi-step recursive)."""
    y = y.astype(float)
    use_lags = [lag for lag in lags if lag < len(y)]
    if not use_lags:
        use_lags = [1]

    def design(series: pd.Series, ex: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
        rows_x, rows_y = [], []
        for t in range(max(use_lags), len(series)):
            feats = [series.iloc[t - lag] for lag in use_lags]
            if ex is not None and t < len(ex):
                feats.extend(ex.iloc[t].tolist())
            rows_x.append(feats)
            rows_y.append(series.iloc[t])
        return np.asarray(rows_x, dtype=float), np.asarray(rows_y, dtype=float)

    ex = exog.reindex(y.index).fillna(0.0) if exog is not None else None
    X_tr, y_tr = design(y, ex)
    model = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.06,
        max_iter=200,
        random_state=seed,
    )
    model.fit(X_tr, y_tr)

    hist = list(y.values.astype(float))
    preds = []
    # future exog
    if ex is not None:
        Xf = make_future_exog(ex, h, country="US")
    else:
        Xf = None

    for step in range(h):
        feats = [hist[-lag] if lag <= len(hist) else hist[0] for lag in use_lags]
        if Xf is not None:
            feats.extend(Xf.iloc[step].tolist())
        pred = float(model.predict(np.asarray(feats, dtype=float).reshape(1, -1))[0])
        pred = max(0.0, pred)
        preds.append(pred)
        hist.append(pred)

    point = np.asarray(preds, dtype=float)
    fitted = model.predict(X_tr)
    sigma = _sigma_from_resid(y_tr - fitted)
    lo, hi = _pi(point, sigma)
    # empirical quantiles from residual distribution
    resid = y_tr - fitted
    q10 = float(np.quantile(resid, 0.1))
    q90 = float(np.quantile(resid, 0.9))
    return AdvForecast(
        name="hgb_lags_exog",
        point=point,
        lower=np.clip(point + q10, 0, None),
        upper=point + q90,
        details=f"HistGradientBoosting lags={use_lags} exog={ex is not None}",
        quantiles={"q10": np.clip(point + q10, 0, None), "q50": point, "q90": point + q90},
    )


def forecast_seasonal_naive(y: pd.Series, h: int, period: int) -> AdvForecast:
    hist = y.astype(float).values
    sp = max(1, min(period, len(hist)))
    point = np.array([hist[-sp + (i % sp)] for i in range(h)], dtype=float)
    if len(hist) > sp:
        sigma = _sigma_from_resid(hist[sp:] - hist[:-sp])
    else:
        sigma = _sigma_from_resid(np.diff(hist)) if len(hist) > 1 else 1.0
    lo, hi = _pi(point, sigma)
    return AdvForecast(
        name=f"seasonal_naive_m{sp}",
        point=np.clip(point, 0, None),
        lower=np.clip(lo, 0, None),
        upper=hi,
        details=f"seasonal naive period={sp}",
    )
