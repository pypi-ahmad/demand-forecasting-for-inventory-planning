"""CUDA-only TimesFM 3.0 zero-shot runner for univariate demand forecasts."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from timesfm3 import TimesFM3Forecaster

TIMESFM3_CHECKPOINT = "google/timesfm-3.0-pytorch"
TIMESFM3_CANDIDATE_NAME = "timesfm_3_zeroshot"
MAX_CONTEXT = 15_360


@dataclass
class TimesFMForecast:
    name: str
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    quantiles: np.ndarray
    details: str


_MODEL_CACHE: TimesFM3Forecaster | None = None


def _require_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TimesFM 3.0 requires CUDA in this project. "
            "Run `uv run python scripts/check_system.py` to diagnose PyTorch."
        )
    return "cuda"


def _require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required to download the TimesFM 3.0 checkpoint. "
            "Accept its non-commercial, non-production license on Hugging Face first."
        )
    return token


def get_timesfm_model() -> TimesFM3Forecaster:
    """Load and cache the licensed TimesFM 3.0 CUDA checkpoint."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        torch.set_float32_matmul_precision("high")
        _MODEL_CACHE = TimesFM3Forecaster.from_pretrained(
            TIMESFM3_CHECKPOINT,
            device=_require_cuda(),
            token=_require_hf_token(),
            per_core_batch_size=4,
        )
    return _MODEL_CACHE


def forecast_timesfm(
    train: pd.Series | np.ndarray,
    h: int,
    *,
    max_context: int | None = None,
) -> TimesFMForecast:
    """Forecast from history only with TimesFM 3.0 on CUDA."""
    arr = np.asarray(train, dtype=np.float32).reshape(-1)
    if int(h) <= 0:
        raise ValueError("h must be positive")
    requested_context = MAX_CONTEXT if max_context is None else int(max_context)
    if requested_context <= 0:
        raise ValueError("max_context must be positive")
    ctx = min(len(arr), requested_context, MAX_CONTEXT)
    if ctx == 0:
        raise ValueError("train must contain at least one value")
    model = get_timesfm_model()
    output = model.predict(
        arr[-ctx:],
        horizon=int(h),
        return_quantiles=True,
        make_positive=True,
    )
    if output.quantiles is None:
        raise RuntimeError("TimesFM 3.0 did not return quantile forecasts")
    quantiles = np.asarray(output.quantiles, dtype=np.float32)
    p = np.clip(np.asarray(output.forecast, dtype=np.float32), 0, None)
    lower = quantiles[:, 0]
    upper = quantiles[:, 8]
    return TimesFMForecast(
        name=TIMESFM3_CANDIDATE_NAME,
        point=p,
        lower=lower,
        upper=upper,
        quantiles=quantiles,
        details=(
            f"TimesFM 3.0 330M zero-shot; checkpoint={TIMESFM3_CHECKPOINT}; "
            f"max_context={ctx}; CUDA; q10-q90"
        ),
    )
