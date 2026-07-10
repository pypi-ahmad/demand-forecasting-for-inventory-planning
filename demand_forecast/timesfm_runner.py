"""TimesFM 2.5 zero-shot runner with production-oriented ForecastConfig."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import timesfm
import torch


@dataclass
class TimesFMForecast:
    name: str
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    quantiles: np.ndarray
    details: str


_MODEL_CACHE: timesfm.TimesFM_2p5_200M_torch | None = None


def get_timesfm_model() -> timesfm.TimesFM_2p5_200M_torch:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        torch.set_float32_matmul_precision("high")
        _MODEL_CACHE = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
    return _MODEL_CACHE


def forecast_timesfm(
    train: pd.Series | np.ndarray,
    h: int,
    *,
    max_context: int | None = None,
) -> TimesFMForecast:
    """Zero-shot forecast; train series is inference context only."""
    arr = np.asarray(train, dtype=np.float32).reshape(-1)
    ctx = int(max_context if max_context is not None else min(len(arr), 2048))
    model = get_timesfm_model()
    model.compile(
        timesfm.ForecastConfig(
            max_context=ctx,
            max_horizon=int(h),
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            per_core_batch_size=32,
        )
    )
    point, quantiles = model.forecast(horizon=int(h), inputs=[arr])
    p = np.clip(point[0], 0, None)
    # SKILL: idx 1 = q10, idx 9 = q90 → ~80% band
    lower = quantiles[0, :, 1]
    upper = quantiles[0, :, 9]
    return TimesFMForecast(
        name="timesfm_2p5_zeroshot",
        point=p,
        lower=lower,
        upper=upper,
        quantiles=quantiles[0],
        details=(
            f"TimesFM 2.5 200M zero-shot; max_context={ctx}; "
            "normalize_inputs; continuous quantile head; infer_is_positive"
        ),
    )
