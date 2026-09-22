# Demand Forecasting for Inventory Planning

Tutorial notebooks and a small Python package for comparing classical demand
forecasts with a zero-shot TimesFM 3.0 baseline. The project is designed for
reproducible, time-ordered evaluation rather than a claim that one model wins
every dataset.

## Current TimesFM runtime

The active foundation-model path uses the official
[`google-research/timesfm` v3.0.2](https://github.com/google-research/timesfm/tree/v3.0.2)
source release and the `google/timesfm-3.0-pytorch` checkpoint. The adapter is
univariate, zero-shot, CUDA-only, and returns a point forecast with q10–q90
intervals. It keeps at most 15,360 recent observations as context.

TimesFM 3.0 source code is Apache-2.0, but its pretrained weights use a
separate non-commercial, non-production license. Do not use this repository’s
TimesFM 3.0 path for commercial or production work. Read the upstream
[license notice](https://github.com/google-research/timesfm/blob/master/README.md#license-notice-for-pretrained-weights)
before downloading weights.

The former benchmark results are not TimesFM 3.0 evidence. They are retained
for audit in [the historical results archive](docs/archive/timesfm-2.5-results.md).
The active notebooks have intentionally not been re-executed with TimesFM 3.0,
so this README makes no current accuracy claim for it.

## Requirements

- Windows 11 with an NVIDIA CUDA-capable GPU and at least 8 GB VRAM
- Python 3.13.13, managed by uv
- Access to the TimesFM 3.0 Hugging Face checkpoint, with `HF_TOKEN` available
  in the process environment
- Network access for the first model download and source datasets

The checked-in lockfile selects CUDA 13.2 PyTorch wheels. CPU fallback is
deliberately disabled: a missing CUDA device is a preflight failure, not a
slower execution mode.

## Install and check

```powershell
git clone https://github.com/pypi-ahmad/demand-forecasting-for-inventory-planning.git
cd demand-forecasting-for-inventory-planning

uv python install 3.13.13
uv sync --locked
uv run python -m ipykernel install --user --name demand-forecast-project
uv run python scripts/check_system.py
```

The preflight checks Python, RAM, disk space, the `timesfm3` API, CUDA-enabled
PyTorch, and available VRAM. For machine-readable output, run:

```powershell
uv run python scripts/check_system.py --json
```

## Forecast workflow

The public adapter remains `forecast_timesfm(train, h, max_context=...)`.
It loads the checkpoint once per process and returns `TimesFMForecast` with:

- `point`: the non-negative median forecast
- `lower` and `upper`: q10 and q90 forecasts
- `quantiles`: all nine deciles from TimesFM 3.0

Classical candidates and the TimesFM baseline are evaluated on the same final
holdout and rolling origins. The production bake-off uses MASE for ranking and
also reports interval coverage and inventory-oriented costs.

## Notebooks

Jupytext percent sources in `notebooks/*.py` are the canonical notebook
sources. Their paired `.ipynb` files are synchronized without stale outputs;
run them yourself to generate TimesFM 3.0 evidence for your data.

```powershell
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_superstore_demand_forecast.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_online_retail_ii_demand_forecast.ipynb
```

After editing a percent source, regenerate its notebook pair:

```powershell
uv run jupytext --to ipynb notebooks/01_superstore_demand_forecast.py
uv run jupytext --to ipynb notebooks/02_online_retail_ii_demand_forecast.py
```

## Repository map

| Path | Purpose |
|---|---|
| `demand_forecast/` | Classical models, metrics, TimesFM 3.0 adapter, and bake-off pipelines |
| `notebooks/` | Tutorial notebooks for Superstore and Online Retail II |
| `scripts/check_system.py` | CUDA and TimesFM 3.0 environment preflight |
| `tests/` | Adapter contract tests that do not download weights |
| `docs/archive/` | Clearly historical benchmark material |

## Troubleshooting

| Symptom | Action |
|---|---|
| CUDA preflight fails | Run `uv sync --locked`, then confirm `torch.cuda.is_available()` is `True`. |
| `HF_TOKEN` missing | Accept the checkpoint terms on Hugging Face and expose `HF_TOKEN` to the current process. |
| TLS certificate verification fails | Configure the organization’s trusted root certificate. Do not disable TLS verification. |
| Model download is slow | Let the first download complete; Hugging Face caches the checkpoint locally. |
| A notebook shows no results | This is expected after migration. Re-execute it before making performance claims. |

## Data and references

The Superstore source, UCI Online Retail II terms, and TimesFM weight terms are
separate from this repository’s MIT code license. Use the original dataset and
model sources when redistributing or publishing work derived from them.

- TimesFM: <https://github.com/google-research/timesfm>
- TimesFM paper: <https://arxiv.org/abs/2310.10688>
- UCI Online Retail II: <https://archive.ics.uci.edu/dataset/502/online+retail+ii>
