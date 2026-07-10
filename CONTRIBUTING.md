# Contributing

Thanks for your interest in **Demand Forecasting for Inventory Planning**. This repo is a tutorial-style, fully executed forecasting project (PyCaret time-series survey + Google TimesFM 2.5). Contributions that improve correctness, reproducibility, docs, or educational clarity are welcome.

## Before you start

1. Read the [README](README.md) (setup, real results, limitations).
2. Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
3. Search [existing issues](https://github.com/pypi-ahmad/demand-forecasting-for-inventory-planning/issues) before opening a new one.

## Development setup

```bash
git clone https://github.com/pypi-ahmad/demand-forecasting-for-inventory-planning.git
cd demand-forecasting-for-inventory-planning

uv python install 3.13.13
uv sync
uv run python -m ipykernel install --user --name demand-forecast-project
uv run python scripts/check_system.py
```

Optional: re-execute notebooks after changes.

```bash
uv run jupyter nbconvert --to notebook --execute \
  notebooks/01_superstore_demand_forecast.ipynb \
  --output 01_superstore_demand_forecast.ipynb \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=demand-forecast-project
```

Jupytext percent sources (`.py`) are the preferred place to edit analysis logic; regenerate `.ipynb` with `uv run jupytext --to ipynb notebooks/<name>.py` when needed.

## What makes a good contribution

| Welcome | Please avoid |
|---------|----------------|
| Bug fixes (data load, API breakage, metrics) | Committing secrets, `.env`, API tokens |
| Doc / README accuracy from real runs | Large binary datasets or model weight dumps |
| Clearer notebook markdown (teach *why*) | Silent metric changes without reporting numbers |
| Faster survey paths that stay evidence-based | Fake or hard-coded “results” in executed cells |
| Tests or small CLI helpers for metrics I/O | Unrelated refactors in the same PR |

**Do not** commit:

- `.venv/`
- `data/*.zip`, `data/*.xlsx` (downloaded at runtime)
- Hugging Face / torch caches
- Personal credentials

## Reporting bugs

Use the **Bug report** issue template. Include:

- OS and `uv run python -V`
- `pycaret`, `timesfm`, `torch` versions (and CUDA yes/no)
- Exact command and full traceback
- Which notebook / cell failed

## Suggesting features

Use the **Feature request** template. Prefer changes that:

- Preserve train/test **time order**
- Keep classical vs TimesFM comparison **fair** (same holdout, same metrics)
- Stay honest about limitations

## Pull requests

1. Fork the repo and create a branch from `main`.
2. Make focused commits (one concern per PR when possible).
3. If behavior or metrics change, re-run the affected notebook and update README numbers if they are advertised as “this run.”
4. Open a PR with:
   - What changed and why  
   - How you verified it (commands + outcome)  
   - Any residual risk  

Maintainers may ask for smaller diffs or additional evidence from a real run.

## Code style (lightweight)

- Python 3.13+, type hints on non-trivial public helpers
- Prefer `pathlib`, f-strings, and explicit errors
- No secrets in the tree
- Prefer `uv add` / `uv lock` over bare `pip install`

## Questions

Open a **Question** issue (or a blank issue if templates do not fit). For dataset licensing, cite UCI / Superstore / TimesFM terms separately from the MIT code license—see the README.
