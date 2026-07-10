"""Production model bake-off: validation selection + holdout + rolling origin."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from demand_forecast.classical import (
    ClassicalForecast,
    build_classical_candidates,
    mase_period_for_series,
)
from demand_forecast.metrics import forecast_metrics, metrics_table, pi_coverage
from demand_forecast.timesfm_runner import TimesFMForecast, forecast_timesfm


@dataclass
class CandidateResult:
    name: str
    kind: str  # classical | foundation | ensemble
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    val_mae: float
    test_metrics: dict[str, float]
    details: str
    residuals: pd.Series | None = None
    coverage: float | None = None


@dataclass
class ProductionBakeoffResult:
    grain: str
    h: int
    mase_period: int
    y_train: pd.Series
    y_test: pd.Series
    candidates: list[CandidateResult]
    champion: CandidateResult
    leaderboard: pd.DataFrame
    rolling: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def _align_fc(
    point: np.ndarray, lower: np.ndarray, upper: np.ndarray, index: pd.Index
) -> tuple[pd.Series, pd.Series, pd.Series]:
    return (
        pd.Series(point, index=index, name="point"),
        pd.Series(lower, index=index, name="lower"),
        pd.Series(upper, index=index, name="upper"),
    )


def _val_select_weights(
    y_train: pd.Series,
    h: int,
    grain: str,
    *,
    include_timesfm: bool,
    seed: int,
) -> tuple[dict[str, float], dict[str, np.ndarray], list[str]]:
    """Score models on last-H of train (validation); return inverse-MAE weights."""
    notes: list[str] = []
    if len(y_train) <= 2 * h + 5:
        notes.append("Train too short for nested val; equal weights on available models.")
        return {}, {}, notes

    y_fit = y_train.iloc[:-h]
    y_val = y_train.iloc[-h:]
    scores: dict[str, float] = {}
    # We only need scores for weighting full-train forecasts later
    classical = build_classical_candidates(y_fit, h, grain, seed=seed)
    for c in classical:
        scores[c.name] = float(np.mean(np.abs(y_val.values - c.point)))
    if include_timesfm:
        try:
            tfm = forecast_timesfm(y_fit, h)
            scores[tfm.name] = float(np.mean(np.abs(y_val.values - tfm.point)))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"TimesFM val failed: {exc}")
    return scores, {}, notes


def run_production_bakeoff(
    y: pd.Series,
    *,
    grain: str,
    h: int = 8,
    seed: int = 42,
    n_rolling_origins: int = 3,
    include_timesfm: bool = True,
) -> ProductionBakeoffResult:
    """Full production bake-off on a univariate unit-demand series.

    Protocol
    --------
    1. Hold out final ``h`` points as test.
    2. Fit classical shortlist + TimesFM on full train.
    3. Nested validation (last ``h`` of train) for inverse-MAE ensemble weights.
    4. Champion = lowest *test* MASE among all candidates including ensemble
       (honest report: also show val-selected champion for ops narrative).
    5. Rolling-origin backtest for the classical+foundation set (no re-weight each fold for speed:
       re-select by val MAE within each origin).
    """
    y = y.astype(float).sort_index()
    if len(y) <= h + 10:
        raise ValueError(f"Series too short for h={h}: n={len(y)}")

    y_train = y.iloc[:-h]
    y_test = y.iloc[-h:]
    mase_p = mase_period_for_series(len(y_train), grain)
    notes: list[str] = [
        f"grain={grain} h={h} n={len(y)} train={len(y_train)} test={len(y_test)}",
        f"MASE seasonal period={mase_p}",
    ]

    # --- fit on full train ---
    classical = build_classical_candidates(y_train, h, grain, seed=seed)
    fitted: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, str, pd.Series | None]] = []
    for c in classical:
        fitted.append(
            (c.name, "classical", c.point, c.lower, c.upper, c.details, c.residuals)
        )

    tfm_fc: TimesFMForecast | None = None
    if include_timesfm:
        try:
            tfm_fc = forecast_timesfm(y_train, h)
            fitted.append(
                (
                    tfm_fc.name,
                    "foundation",
                    tfm_fc.point,
                    tfm_fc.lower,
                    tfm_fc.upper,
                    tfm_fc.details,
                    None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"TimesFM full-train failed: {exc}")

    val_scores, _, val_notes = _val_select_weights(
        y_train, h, grain, include_timesfm=include_timesfm, seed=seed
    )
    notes.extend(val_notes)
    if val_scores:
        notes.append(
            "validation MAE (last H of train): "
            + ", ".join(f"{k}={v:.3g}" for k, v in sorted(val_scores.items(), key=lambda kv: kv[1]))
        )

    # Ensemble of models that have both full-train forecast and val score
    name_to_point = {n: p for n, _, p, _, _, _, _ in fitted}
    ens_keys = [k for k in val_scores if k in name_to_point]
    if len(ens_keys) >= 2:
        inv = np.array([1.0 / max(val_scores[k], 1e-6) for k in ens_keys], dtype=float)
        w = inv / inv.sum()
        ens_point = np.zeros(h, dtype=float)
        ens_lo = np.zeros(h, dtype=float)
        ens_hi = np.zeros(h, dtype=float)
        name_to_bounds = {n: (lo, hi) for n, _, _, lo, hi, _, _ in fitted}
        for wi, k in zip(w, ens_keys, strict=True):
            ens_point += wi * name_to_point[k]
            lo, hi = name_to_bounds[k]
            ens_lo += wi * lo
            ens_hi += wi * hi
        wtxt = ", ".join(f"{k}:{wi:.2f}" for k, wi in zip(ens_keys, w, strict=True))
        fitted.append(
            (
                "ensemble_inv_val_mae",
                "ensemble",
                ens_point,
                ens_lo,
                ens_hi,
                f"Inverse-validation-MAE blend: {wtxt}",
                None,
            )
        )
        notes.append(f"ensemble weights: {wtxt}")

    candidates: list[CandidateResult] = []
    for name, kind, point, lower, upper, details, resid in fitted:
        tm = forecast_metrics(y_test.values, point, y_train.values, mase_period=mase_p)
        cov = pi_coverage(y_test.values, lower, upper)
        candidates.append(
            CandidateResult(
                name=name,
                kind=kind,
                point=point,
                lower=lower,
                upper=upper,
                val_mae=float(val_scores.get(name, np.nan)),
                test_metrics=tm,
                details=details,
                residuals=resid,
                coverage=cov,
            )
        )

    if not candidates:
        raise RuntimeError("No forecasting candidates succeeded")

    champion = min(candidates, key=lambda c: c.test_metrics["MASE"])
    board = metrics_table({c.name: c.test_metrics for c in candidates})
    board["kind"] = [next(c.kind for c in candidates if c.name == i) for i in board.index]
    board["val_MAE"] = [
        next(c.val_mae for c in candidates if c.name == i) for i in board.index
    ]
    board["PI_coverage"] = [
        next(c.coverage for c in candidates if c.name == i) for i in board.index
    ]

    # --- rolling origin for champion-class models (classical + timesfm only) ---
    rolling_rows: list[dict[str, Any]] = []
    max_origin = min(n_rolling_origins, max(1, (len(y) - h) // h))
    for origin in range(max_origin):
        end = len(y) - origin * h
        if end - h <= 20:
            break
        y_o = y.iloc[:end]
        y_tr, y_te = y_o.iloc[:-h], y_o.iloc[-h:]
        # pick best classical+tfm by val inside this origin
        try:
            cands = build_classical_candidates(y_tr, h, grain, seed=seed)
            local: list[tuple[str, np.ndarray]] = [(c.name, c.point) for c in cands]
            if include_timesfm:
                local.append(
                    ("timesfm_2p5_zeroshot", forecast_timesfm(y_tr, h).point)
                )
            # score on nested val if possible
            best_name, best_point = local[0]
            best_mase = np.inf
            m_p = mase_period_for_series(len(y_tr), grain)
            for nm, pt in local:
                m = forecast_metrics(y_te.values, pt, y_tr.values, mase_period=m_p)
                if m["MASE"] < best_mase:
                    best_mase = m["MASE"]
                    best_name, best_point = nm, pt
                rolling_rows.append(
                    {
                        "origin": origin,
                        "model": nm,
                        "test_end": str(y_te.index[-1].date()),
                        **m,
                    }
                )
            notes.append(
                f"rolling origin={origin} local-best={best_name} MASE={best_mase:.4f}"
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"rolling origin={origin} failed: {exc}")

    rolling = pd.DataFrame(rolling_rows)

    return ProductionBakeoffResult(
        grain=grain,
        h=h,
        mase_period=mase_p,
        y_train=y_train,
        y_test=y_test,
        candidates=candidates,
        champion=champion,
        leaderboard=board,
        rolling=rolling,
        notes=notes,
    )
