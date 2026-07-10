"""End-to-end advanced forecasting pipeline (v3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from demand_forecast.advanced.ensemble import blend_points, smart_stack_weights
from demand_forecast.advanced.evaluation import rolling_origin_evaluate, summarize_rolling
from demand_forecast.advanced.features import build_calendar_frame
from demand_forecast.advanced.hierarchy import bottom_up_forecast, build_bottom_panel
from demand_forecast.advanced.inventory import (
    asymmetric_cost,
    quantile_order_quantity,
    safety_buffer_from_band,
)
from demand_forecast.advanced.models_exog import (
    AdvForecast,
    forecast_hw_log,
    forecast_hw_mul,
    forecast_ml_lags,
    forecast_sarimax_exog,
    forecast_seasonal_naive,
)
from demand_forecast.metrics import forecast_metrics, metrics_table
from demand_forecast.timesfm_runner import forecast_timesfm


@dataclass
class AdvancedRunResult:
    dataset: str
    grain: str
    h: int
    y_train: pd.Series
    y_test: pd.Series
    leaderboard: pd.DataFrame
    inventory_table: pd.DataFrame
    champion_name: str
    champion_point: np.ndarray
    champion_lower: np.ndarray
    champion_upper: np.ndarray
    order_qty_sl90: np.ndarray
    safety_buffer: np.ndarray
    hierarchy_total: np.ndarray | None
    rolling_summary: dict[str, pd.DataFrame]
    notes: list[str] = field(default_factory=list)
    forecasts: dict[str, AdvForecast] = field(default_factory=dict)


def _best_period(n: int) -> int:
    """Longest seasonal period with ≥2 full cycles (fallback chain)."""
    for p in (52, 26, 13, 8, 4):
        if n >= 2 * p + 2:
            return p
    return max(2, n // 4)


def _select_period_by_val(y_train: pd.Series, h: int) -> int:
    """Pick HW seasonal period by nested validation MAE (production hyperparam)."""
    if len(y_train) <= 2 * h + 10:
        return _best_period(len(y_train))
    y_fit, y_val = y_train.iloc[:-h], y_train.iloc[-h:]
    candidates = [p for p in (4, 8, 13, 26, 52) if len(y_fit) >= 2 * p + 2]
    if not candidates:
        return _best_period(len(y_train))
    best_p, best_mae = candidates[0], float("inf")
    for p in candidates:
        try:
            fc = forecast_hw_mul(y_fit, h, period=p)
            mae = float(np.mean(np.abs(y_val.values - fc.point)))
            if mae < best_mae:
                best_mae, best_p = mae, p
        except Exception:
            continue
    return best_p


def _mase_p(n: int) -> int:
    return 52 if n > 60 else 4


def run_advanced_pipeline(
    y_total: pd.Series,
    *,
    dataset: str,
    transactions: pd.DataFrame | None = None,
    date_col: str = "Order Date",
    qty_col: str = "Quantity",
    hierarchy_cols: list[str] | None = None,
    promo_series: pd.Series | None = None,
    country: str = "US",
    h: int = 8,
    n_rolling: int = 6,
    underage_cost: float = 4.0,
    overage_cost: float = 1.0,
    seed: int = 42,
) -> AdvancedRunResult:
    """Run advanced stack on weekly unit totals (+ optional hierarchy from transactions)."""
    notes: list[str] = []
    y = y_total.astype(float).sort_index()
    y_train, y_test = y.iloc[:-h], y.iloc[-h:]
    mase_period = _mase_p(len(y_train))
    period = _select_period_by_val(y_train, h)
    notes.append(
        f"dataset={dataset} n={len(y)} train={len(y_train)} test={h} "
        f"val_selected_period={period} mase_period={mase_period}"
    )

    # --- exogenous calendar (+ optional promo) ---
    cal = build_calendar_frame(y_train.index.union(y_test.index), country=country)
    if promo_series is not None:
        cal = cal.join(promo_series.rename("promo_proxy"), how="left").fillna(0.0)
    exog_train = cal.reindex(y_train.index).fillna(0.0)

    forecasts: dict[str, AdvForecast] = {}

    # 1) seasonal naive
    forecasts["seasonal_naive"] = forecast_seasonal_naive(y_train, h, period)

    # 2) multiplicative HW for *all* feasible periods (period is a hyperparameter)
    period_candidates = [p for p in (4, 8, 13, 26, 52) if len(y_train) >= 2 * p + 2]
    if not period_candidates:
        period_candidates = [period]
    for p in period_candidates:
        try:
            forecasts[f"hw_mul_m{p}"] = forecast_hw_mul(y_train, h, period=p)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"hw_mul_m{p} failed: {exc}")
    # alias val-selected period for readability
    if f"hw_mul_m{period}" in forecasts:
        forecasts["hw_mul"] = forecasts[f"hw_mul_m{period}"]

    # 3) log1p + HW (variance stabilization) at val-selected period
    try:
        forecasts["hw_log1p"] = forecast_hw_log(y_train, h, period=period, seasonal="add")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"hw_log1p failed: {exc}")

    # 4) SARIMAX + exog
    try:
        forecasts["sarimax_exog"] = forecast_sarimax_exog(
            y_train,
            exog_train,
            h,
            order=(1, 0, 1),
            seasonal_order=(0, 1, 1, min(4, period)),
            country=country,
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"sarimax_exog failed: {exc}")

    # 5) ML lags + exog
    try:
        forecasts["hgb_lags_exog"] = forecast_ml_lags(
            y_train, exog_train, h, seed=seed
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"hgb_lags_exog failed: {exc}")

    # 6) TimesFM zero-shot (no jax xreg required)
    try:
        tfm = forecast_timesfm(y_train, h)
        forecasts["timesfm_zeroshot"] = AdvForecast(
            name="timesfm_zeroshot",
            point=tfm.point,
            lower=tfm.lower,
            upper=tfm.upper,
            details=tfm.details + " | XReg optional (needs timesfm[xreg]+jax; not required here)",
            quantiles={"q10": tfm.lower, "q50": tfm.point, "q90": tfm.upper},
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"timesfm failed: {exc}")

    # 7) Hierarchy bottom-up (if transactions provided)
    hierarchy_total = None
    if transactions is not None and hierarchy_cols:
        try:
            panel = build_bottom_panel(
                transactions,
                date_col=date_col,
                qty_col=qty_col,
                group_cols=hierarchy_cols,
                min_total_qty=80.0,
            )
            # align panel to train end
            panel_tr = panel.loc[: y_train.index.max()]
            hier = bottom_up_forecast(panel_tr, h, seasonal="mul", top_k=8)
            hierarchy_total = hier.total_bottom_up
            # Scale bottom-up to recent total level (simple reconciliation)
            recent = float(y_train.tail(min(8, len(y_train))).mean())
            bu_level = float(np.mean(hierarchy_total)) if np.mean(hierarchy_total) > 0 else 1.0
            scale = recent / bu_level
            # Keep scale near 1 unless hierarchy drifted badly
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            scale = float(np.clip(scale, 0.25, 4.0))
            hierarchy_total = hierarchy_total * scale
            forecasts["hierarchy_bottom_up"] = AdvForecast(
                name="hierarchy_bottom_up",
                point=np.clip(hierarchy_total, 0, None),
                lower=np.clip(hierarchy_total * 0.85, 0, None),
                upper=hierarchy_total * 1.15,
                details="; ".join(hier.notes)
                + f" | bottoms={len(hier.series_used)} groups={hierarchy_cols} scale={scale:.3f}",
            )
            notes.extend(hier.notes)
            notes.append(f"hierarchy level-scale={scale:.3f} (align to recent total mean)")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"hierarchy failed: {exc}")

    # --- nested validation for ensemble (last h of train) ---
    val_mae: dict[str, float] = {}
    if len(y_train) > 2 * h + 10:
        y_fit, y_val = y_train.iloc[:-h], y_train.iloc[-h:]
        # re-fit lightweight candidates on y_fit
        val_points: dict[str, np.ndarray] = {}
        # score key models + best hw period on nested val
        builders: dict[str, Any] = {
            "seasonal_naive": lambda: forecast_seasonal_naive(y_fit, h, period).point,
            "hw_log1p": lambda: forecast_hw_log(y_fit, h, period=period).point,
            "timesfm_zeroshot": lambda: forecast_timesfm(y_fit, h).point,
        }
        for p in period_candidates:
            builders[f"hw_mul_m{p}"] = (
                lambda p=p: forecast_hw_mul(y_fit, h, period=p).point
            )
        for name, builder in builders.items():
            try:
                val_points[name] = builder()
                val_mae[name] = float(
                    np.mean(np.abs(y_val.values - val_points[name]))
                )
            except Exception:
                continue
        if len(val_mae) >= 2:
            w = smart_stack_weights(val_mae, max_models=3)
            # map stack keys to full-train forecasts
            full_pts = {}
            for k in w:
                if k in forecasts:
                    full_pts[k] = forecasts[k].point
                elif k == "hw_mul" and "hw_mul" in forecasts:
                    full_pts[k] = forecasts["hw_mul"].point
            if full_pts:
                blended = blend_points(full_pts, w)
                lo = blend_points({k: forecasts[k].lower for k in full_pts}, w)
                hi = blend_points({k: forecasts[k].upper for k in full_pts}, w)
                # re-normalize weights to keys actually blended
                w_used = {k: w[k] for k in full_pts}
                s = sum(w_used.values()) or 1.0
                w_used = {k: v / s for k, v in w_used.items()}
                forecasts["smart_stack"] = AdvForecast(
                    name="smart_stack",
                    point=np.clip(blended, 0, None),
                    lower=lo,
                    upper=hi,
                    details=(
                        "inverse-val-MAE stack weights="
                        + str({k: round(v, 3) for k, v in w_used.items()})
                    ),
                )
                notes.append(f"smart_stack weights={w_used}")

    # --- holdout metrics + inventory scores ---
    metric_rows: dict[str, dict[str, float]] = {}
    inv_rows: list[dict[str, Any]] = []
    for name, fc in forecasts.items():
        m = forecast_metrics(
            y_test.values, fc.point, y_train.values, mase_period=mase_period
        )
        metric_rows[name] = m
        inv = asymmetric_cost(
            y_test.values,
            fc.point,
            underage_cost=underage_cost,
            overage_cost=overage_cost,
        )
        # also score quantile order at SL=0.9
        oq = quantile_order_quantity(fc.point, fc.lower, fc.upper, service_level=0.9)
        inv90 = asymmetric_cost(
            y_test.values, oq, underage_cost=underage_cost, overage_cost=overage_cost
        )
        inv_rows.append(
            {
                "model": name,
                "under_units": inv.under_units,
                "over_units": inv.over_units,
                "asymmetric_cost": inv.asymmetric_cost,
                "service_hit_rate": inv.service_level_hit_rate,
                "asymmetric_cost_sl90_order": inv90.asymmetric_cost,
                "under_units_sl90": inv90.under_units,
                "MAE": m["MAE"],
                "MASE": m["MASE"],
                "MAPE": m["MAPE"],
            }
        )

    leaderboard = metrics_table(metric_rows)
    inventory_table = (
        pd.DataFrame(inv_rows).set_index("model").sort_values("asymmetric_cost")
    )

    # Dual gate: among models within 15% of best MASE, pick lowest inventory cost.
    # Prevents "over-forecast everything" from winning on underage alone.
    best_mase = float(leaderboard["MASE"].min())
    mase_ok = leaderboard.index[leaderboard["MASE"] <= 1.15 * best_mase].tolist()
    eligible = inventory_table.loc[inventory_table.index.intersection(mase_ok)]
    if eligible.empty:
        champion_name = str(leaderboard.index[0])
        notes.append("champion=best_MASE (no model passed MASE eligibility for cost gate)")
    else:
        champion_name = str(eligible.sort_values("asymmetric_cost").index[0])
        notes.append(
            f"champion gate: MASE<={1.15 * best_mase:.4f} then min asymmetric_cost "
            f"among {list(eligible.index)}"
        )
    accuracy_champion = str(leaderboard.index[0])
    if accuracy_champion != champion_name:
        notes.append(f"accuracy_champion_by_MASE={accuracy_champion}")
    champ = forecasts[champion_name]
    order_qty = quantile_order_quantity(
        champ.point, champ.lower, champ.upper, service_level=0.9
    )
    safety = safety_buffer_from_band(champ.point, champ.upper)

    # --- rolling origin for key models ---
    rolling_summary: dict[str, pd.DataFrame] = {}

    def _roll(name: str, fn) -> None:
        try:
            rdf = rolling_origin_evaluate(
                y,
                fn,
                h=h,
                n_origins=n_rolling,
                min_train=max(40, 2 * period + 5),
                mase_period=mase_period,
                underage_cost=underage_cost,
                overage_cost=overage_cost,
            )
            rolling_summary[name] = rdf
        except Exception as exc:  # noqa: BLE001
            notes.append(f"rolling {name} failed: {exc}")

    _roll("hw_mul", lambda tr, hh: forecast_hw_mul(tr, hh, period=_best_period(len(tr))).point)
    _roll("seasonal_naive", lambda tr, hh: forecast_seasonal_naive(tr, hh, _best_period(len(tr))).point)
    _roll("timesfm_zeroshot", lambda tr, hh: forecast_timesfm(tr, hh).point)

    notes.append(
        f"champion_by_asymmetric_cost={champion_name} "
        f"cost={inventory_table.loc[champion_name, 'asymmetric_cost']:.1f}"
    )

    return AdvancedRunResult(
        dataset=dataset,
        grain="weekly",
        h=h,
        y_train=y_train,
        y_test=y_test,
        leaderboard=leaderboard,
        inventory_table=inventory_table,
        champion_name=champion_name,
        champion_point=champ.point,
        champion_lower=champ.lower,
        champion_upper=champ.upper,
        order_qty_sl90=order_qty,
        safety_buffer=safety,
        hierarchy_total=hierarchy_total,
        rolling_summary=rolling_summary,
        notes=notes,
        forecasts=forecasts,
    )
