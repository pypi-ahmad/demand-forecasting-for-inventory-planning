"""Small diverse ensembles with rolling-validation weights."""

from __future__ import annotations

import numpy as np


def inverse_error_weights(errors: dict[str, float], eps: float = 1e-6) -> dict[str, float]:
    inv = {k: 1.0 / max(float(v), eps) for k, v in errors.items()}
    s = sum(inv.values())
    return {k: v / s for k, v in inv.items()}


def blend_points(
    points: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    keys = [k for k in weights if k in points]
    if not keys:
        raise ValueError("No overlapping keys for blend")
    out = np.zeros_like(points[keys[0]], dtype=float)
    wsum = 0.0
    for k in keys:
        w = float(weights[k])
        out += w * np.asarray(points[k], dtype=float)
        wsum += w
    return out / wsum


def smart_stack_weights(
    val_mae: dict[str, float],
    *,
    max_models: int = 3,
) -> dict[str, float]:
    """Keep only the best ``max_models`` by val MAE, then inverse-error weight."""
    ranked = sorted(val_mae.items(), key=lambda kv: kv[1])[:max_models]
    return inverse_error_weights(dict(ranked))
