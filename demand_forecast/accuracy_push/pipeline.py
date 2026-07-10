"""v4 accuracy-push pipeline: hierarchy + multi-window model selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from demand_forecast.accuracy_push.hierarchy_safe import (
    blend,
    bottom_up_total,
    volume_scale,
)
from demand_forecast.advanced.features import (
    aggregate_promo_proxy,
    build_calendar_frame,
)
from demand_forecast.advanced.inventory import asymmetric_cost, quantile_order_quantity
from demand_forecast.metrics import forecast_metrics, metrics_table
from demand_forecast.timesfm_runner import forecast_timesfm


@dataclass
class AccuracyPushResult:
    dataset: str
    h: int
    y_train: pd.Series
    y_test: pd.Series
    leaderboard: pd.DataFrame
    multiwindow_scores: pd.DataFrame
    champion_name: str
    champion_point: np.ndarray
    champion_lower: np.ndarray
    champion_upper: np.ndarray
    order_qty_sl90: np.ndarray
    notes: list[str] = field(default_factory=list)
    forecasts: dict[str, np.ndarray] = field(default_factory=dict)


def _hw_mul(train: pd.Series, h: int, period: int) -> np.ndarray:
    y = train.astype(float).clip(lower=1e-3)
    if len(y) < 2 * period + 2:
        sp = max(1, min(period, len(y)))
        hist = y.values
        return np.array([hist[-sp + (i % sp)] for i in range(h)], dtype=float)
    model = ExponentialSmoothing(
        y,
        trend="add",
        seasonal="mul",
        seasonal_periods=period,
        initialization_method="estimated",
    ).fit(optimized=True)
    return np.clip(np.asarray(model.forecast(h), dtype=float), 0, None)


def _sigma_pi(train: pd.Series, point: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Rough residual PI from seasonal-naive residuals on train."""
    hist = train.astype(float).values
    sp = max(1, min(period, len(hist) // 2))
    if len(hist) <= sp:
        sigma = float(np.std(hist)) if len(hist) else 1.0
    else:
        sigma = float(np.std(hist[sp:] - hist[:-sp]))
    sigma = sigma if sigma > 0 else 1.0
    z = 1.28155
    return point - z * sigma, point + z * sigma


def _origin_splits(y: pd.Series, h: int, n_origins: int, min_train: int) -> list[tuple[pd.Series, pd.Series]]:
    out = []
    for origin in range(n_origins):
        end = len(y) - origin * h
        if end - h < min_train:
            break
        out.append((y.iloc[: end - h], y.iloc[end - h : end]))
    return out


def run_accuracy_push(
    y: pd.Series,
    transactions: pd.DataFrame,
    *,
    dataset: str,
    date_col: str,
    qty_col: str,
    hierarchy_specs: list[list[str]],
    promo_col: str | None = None,
    country: str = "US",
    h: int = 8,
    n_origins: int = 6,
    seed: int = 42,
) -> AccuracyPushResult:
    """Fit candidate models; pick champion by mean MAE across rolling origins; score final holdout."""
    notes: list[str] = []
    y = y.astype(float).sort_index()
    y_train, y_test = y.iloc[:-h], y.iloc[-h:]
    notes.append(f"dataset={dataset} n={len(y)} train={len(y_train)} test={h} origins={n_origins}")

    # --- define candidate generators: name -> fn(train_series, h) -> point ---
    # period defaults
    def periods_for(n: int) -> list[int]:
        return [p for p in (52, 26, 13, 8, 4) if n >= 2 * p + 2] or [4]

    candidates: dict[str, Callable[[pd.Series, int], np.ndarray]] = {}

    # univariate HW for each period
    for p in periods_for(len(y_train)):
        candidates[f"hw_mul_m{p}"] = lambda tr, hh, p=p: _hw_mul(tr, hh, p)

    # TimesFM
    def _tfm(tr: pd.Series, hh: int) -> np.ndarray:
        return forecast_timesfm(tr, hh).point

    candidates["timesfm_zeroshot"] = _tfm

    # hierarchy bottom-up per spec
    for groups in hierarchy_specs:
        gname = "bu_" + "_".join(g.lower() for g in groups)

        def _bu(tr: pd.Series, hh: int, groups=groups, gname=gname) -> np.ndarray:
            p = periods_for(len(tr))[0]
            res = bottom_up_total(
                transactions,
                date_col=date_col,
                qty_col=qty_col,
                group_cols=groups,
                train_index=tr.index,
                h=hh,
                period=p,
            )
            return res.point

        candidates[gname] = _bu

        # volume-scaled to univariate HW best-period reference
        def _bu_scaled(tr: pd.Series, hh: int, groups=groups, gname=gname) -> np.ndarray:
            p = periods_for(len(tr))[0]
            base = _hw_mul(tr, hh, p)
            res = bottom_up_total(
                transactions,
                date_col=date_col,
                qty_col=qty_col,
                group_cols=groups,
                train_index=tr.index,
                h=hh,
                period=p,
            )
            return volume_scale(res.point, base)

        candidates[gname + "_volscaled"] = _bu_scaled

        # blend HW + bottom-up (alpha=0.5 default; refined below on origins)
        def _bu_blend(tr: pd.Series, hh: int, groups=groups) -> np.ndarray:
            p = periods_for(len(tr))[0]
            base = _hw_mul(tr, hh, p)
            res = bottom_up_total(
                transactions,
                date_col=date_col,
                qty_col=qty_col,
                group_cols=groups,
                train_index=tr.index,
                h=hh,
                period=p,
            )
            scaled = volume_scale(res.point, base)
            return blend(base, scaled, 0.5)

        candidates[gname + "_blend50"] = _bu_blend

    # residual calendar correction on top of best univariate period (52 if possible)
    uni_p = periods_for(len(y_train))[0]

    def _hw_resid_calendar(tr: pd.Series, hh: int) -> np.ndarray:
        from sklearn.linear_model import Ridge

        p = periods_for(len(tr))[0]
        base = _hw_mul(tr, hh, p)
        # fit residual model on in-sample
        yfit = tr.clip(lower=1e-3)
        try:
            model = ExponentialSmoothing(
                yfit,
                trend="add",
                seasonal="mul",
                seasonal_periods=p,
                initialization_method="estimated",
            ).fit(optimized=True)
            fitted = np.asarray(model.fittedvalues, dtype=float)
            resid = tr.values - fitted
            cal = build_calendar_frame(tr.index, country=country)
            if promo_col and promo_col in transactions.columns:
                promo = aggregate_promo_proxy(
                    transactions, date_col=date_col, value_col=promo_col
                ).reindex(tr.index).fillna(0.0)
                cal = cal.join(promo.rename("promo"), how="left").fillna(0.0)
            mask = np.isfinite(resid)
            ridge = Ridge(alpha=1.0)
            ridge.fit(cal.values[mask], resid[mask])
            # future calendar for forecast horizon
            last = tr.index[-1]
            future_idx = pd.date_range(last, periods=hh + 1, freq="W-SUN")[1:]
            cal_f = build_calendar_frame(future_idx, country=country)
            if "promo" in cal.columns:
                cal_f["promo"] = 0.0
            for c in cal.columns:
                if c not in cal_f.columns:
                    cal_f[c] = 0.0
            cal_f = cal_f[cal.columns]
            adj = ridge.predict(cal_f.values)
            return np.clip(base + adj, 0, None)
        except Exception:
            return base

    candidates["hw_mul_plus_calendar_resid"] = _hw_resid_calendar

    # --- multi-window scores ---
    splits = _origin_splits(y, h, n_origins, min_train=max(40, 2 * uni_p + 5))
    rows = []
    for origin, (tr, te) in enumerate(splits):
        for name, fn in candidates.items():
            try:
                point = np.asarray(fn(tr, h), dtype=float)
                m = forecast_metrics(
                    te.values,
                    point,
                    tr.values,
                    mase_period=52 if len(tr) > 60 else 4,
                )
                inv = asymmetric_cost(te.values, point, underage_cost=4.0, overage_cost=1.0)
                rows.append(
                    {
                        "origin": origin,
                        "model": name,
                        "test_end": str(te.index[-1].date()),
                        **m,
                        "asymmetric_cost": inv.asymmetric_cost,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {"origin": origin, "model": name, "error": str(exc), "MAE": np.nan, "MASE": np.nan}
                )

    mw = pd.DataFrame(rows)
    # mean MAE across origins for ranking (ignore failed)
    score = (
        mw.dropna(subset=["MAE"])
        .groupby("model")[["MAE", "MASE", "MAPE", "asymmetric_cost"]]
        .mean()
        .sort_values("MAE")
    )
    notes.append("multi-window ranking by mean MAE across origins")
    notes.append("top3_mw=" + ", ".join(f"{i}:{score.loc[i, 'MAE']:.2f}" for i in score.head(3).index))

    # champion = best multi-window mean MAE
    champion_name = str(score.index[0]) if len(score) else "hw_mul_m52"
    notes.append(f"champion_by_mean_origin_MAE={champion_name}")

    # final fit on full train
    final_points: dict[str, np.ndarray] = {}
    for name, fn in candidates.items():
        try:
            final_points[name] = np.asarray(fn(y_train, h), dtype=float)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"final fit failed {name}: {exc}")

    # holdout leaderboard
    metric_rows = {}
    for name, point in final_points.items():
        metric_rows[name] = forecast_metrics(
            y_test.values,
            point,
            y_train.values,
            mase_period=52 if len(y_train) > 60 else 4,
        )
        inv = asymmetric_cost(y_test.values, point)
        metric_rows[name]["asymmetric_cost"] = inv.asymmetric_cost
        metric_rows[name]["under_units"] = inv.under_units
        metric_rows[name]["over_units"] = inv.over_units

    leaderboard = pd.DataFrame(metric_rows).T
    # sort by MAE for accuracy-push narrative
    if "MAE" in leaderboard.columns:
        leaderboard = leaderboard.sort_values("MAE")

    # also compute holdout-best for honesty
    if len(leaderboard):
        holdout_best = str(leaderboard.index[0])
        notes.append(f"holdout_best_MAE={holdout_best} MAE={leaderboard.iloc[0]['MAE']:.4f}")
        if holdout_best != champion_name:
            notes.append(
                "NOTE: multi-window champion differs from single holdout best "
                f"({champion_name} vs {holdout_best})"
            )

    # use multi-window champion for reported point; fall back to holdout best if missing
    if champion_name not in final_points:
        champion_name = str(leaderboard.index[0])
    point = final_points[champion_name]
    # PI from residual scale of seasonal naive
    lo, hi = _sigma_pi(y_train, point, uni_p)
    order = quantile_order_quantity(point, lo, hi, service_level=0.9)

    return AccuracyPushResult(
        dataset=dataset,
        h=h,
        y_train=y_train,
        y_test=y_test,
        leaderboard=leaderboard,
        multiwindow_scores=score,
        champion_name=champion_name,
        champion_point=point,
        champion_lower=lo,
        champion_upper=hi,
        order_qty_sl90=order,
        notes=notes,
        forecasts=final_points,
    )
