# Python reference

The root package exposes these baseline functions:

```python
from demand_forecast import forecast_metrics, metrics_table, run_production_bakeoff
```

Pass unit-demand series in time order.

## `run_production_bakeoff`

```python
run_production_bakeoff(
    y: pandas.Series,
    *,
    grain: str,
    h: int = 8,
    seed: int = 42,
    n_rolling_origins: int = 3,
    include_timesfm: bool = True,
) -> ProductionBakeoffResult
```

Runs the classical shortlist, optional TimesFM 3.0 forecast, validation-weighted
ensemble, final holdout scoring, and rolling-origin evaluation.

| Parameter | Meaning |
|---|---|
| `y` | Numeric `pandas.Series`; it is sorted and converted to float. |
| `grain` | Use `"weekly"` or `"daily"` to select the seasonal-period helpers. |
| `h` | Final holdout and forecast horizon length. The series needs more than `h + 10` points. |
| `seed` | AutoARIMA random state. |
| `n_rolling_origins` | Maximum number of expanding rolling origins. |
| `include_timesfm` | Adds the CUDA-only TimesFM candidate when `True`. |

`ProductionBakeoffResult` contains `y_train`, `y_test`, the candidate list,
the MASE-sorted `leaderboard`, rolling-origin rows, notes, and `champion`.
`champion` is a `CandidateResult` with point and interval arrays, validation
MAE, holdout metrics, interval coverage, and model details.

```python
result = run_production_bakeoff(series, grain="weekly", h=8)
print(result.champion.name)
print(result.leaderboard[["MAE", "MASE", "PI_coverage"]])
```

## `forecast_timesfm`

```python
from demand_forecast.timesfm_runner import forecast_timesfm

forecast_timesfm(train, h, *, max_context=None) -> TimesFMForecast
```

Runs the `google/timesfm-3.0-pytorch` checkpoint on CUDA. `train` may be a
`pandas.Series` or `numpy.ndarray`; it is flattened to one float32 series.
`h` and `max_context` must be positive. Context is capped at 15,360 values.

`TimesFMForecast` fields:

| Field | Meaning |
|---|---|
| `name` | `timesfm_3_zeroshot` |
| `point` | Non-negative median forecast, shape `(h,)` |
| `lower` | q10 forecast, shape `(h,)` |
| `upper` | q90 forecast, shape `(h,)` |
| `quantiles` | Nine TimesFM deciles, shape `(h, 9)` |
| `details` | Checkpoint, context, CUDA, and interval metadata |

The function fails early when CUDA is unavailable, `HF_TOKEN` is absent, or
the checkpoint cannot return quantiles. It does not fine-tune the model or
accept covariates.

## Metrics

```python
forecast_metrics(y_true, y_pred, y_train, *, mase_period=1) -> dict[str, float]
metrics_table(rows) -> pandas.DataFrame
```

`forecast_metrics` returns `MAE`, `RMSE`, `MAPE`, `sMAPE`, `MASE`, and `bias`.
MASE uses a seasonal-naive scale from `y_train`, capped to a feasible period and
falling back to a one-step scale when needed. Bias is mean predicted demand
minus mean actual demand; a negative value is under-forecasting on average.

`metrics_table` accepts `{model_name: metric_dict}` and returns a table sorted
by ascending MASE. `pi_coverage` is available from `demand_forecast.metrics`
for the fraction of actuals inside an interval.

## Environment preflight

```powershell
uv run python scripts/check_system.py
uv run python scripts/check_system.py --json
```

The preflight checks RAM, CUDA availability and VRAM, disk space, Python,
`timesfm3`, and `torch`. It returns a non-zero exit code when a required check
fails. Optional `--num-series`, `--context-length`, `--horizon`, and
`--batch-size` arguments print a memory estimate alongside the main report.
