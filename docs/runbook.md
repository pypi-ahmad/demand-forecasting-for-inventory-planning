# Operator runbook

This runbook covers reproducing the baseline and diagnosing its runtime. It
applies to the CUDA-only TimesFM 3.0 path on Windows.

## Prepare the environment

```powershell
uv python install 3.13.13
uv sync --locked
uv run python -m ipykernel install --user --name demand-forecast-project
uv run python scripts/check_system.py --json
```

Do not replace `uv sync --locked` with an unlocked install when reproducing the
checked-in results. The expected preflight result is `passed: true`, GPU mode,
and a recommended batch size of four.

Before the first TimesFM download, accept the upstream checkpoint terms and
make `HF_TOKEN` available to the current process. Do not put the token in a
notebook, source file, or commit.

## Execute the verified notebooks

```powershell
$env:PYTHONUTF8 = "1"
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/01_superstore_demand_forecast.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/02_online_retail_ii_demand_forecast.ipynb
```

Notebook 02 first tries `ucimlrepo` for Online Retail II, then falls back to
the official UCI ZIP when that dataset is unavailable through the Python API.
The source ZIP is cached under `data/` and is ignored by Git.

The resulting leaderboards are written to:

- `data/results/superstore_production_metrics.csv`
- `data/results/online_retail_ii_production_metrics.csv`

## Interpret a run

1. Read `leaderboard` in ascending MASE order for the final holdout ranking.
2. Check `PI_coverage` before using a q10–q90 interval for a service buffer.
3. Compare rolling-origin rows with the holdout; do not make a broad model
   claim from one cutoff.
4. Treat the notebooks’ inventory commentary as directional, not a complete
   safety-stock policy.

## Troubleshooting

| Symptom | Check | Resolution |
|---|---|---|
| Preflight reports no CUDA GPU | `uv run python -c "import torch; print(torch.cuda.is_available())"` | Re-sync the locked environment and verify the NVIDIA driver. CPU execution is intentionally unsupported. |
| `HF_TOKEN is required` | Confirm the process environment contains `HF_TOKEN` | Accept the checkpoint terms and relaunch the shell or notebook kernel if the token was just added. |
| TLS certificate verification error | Check corporate or local certificate configuration | Install the appropriate trusted root certificate. Do not disable TLS validation. |
| `No such kernel named demand-forecast-project` | `uv run jupyter kernelspec list` | Re-run the `ipykernel install` command above. |
| `UnicodeEncodeError` from a direct notebook-source run | Confirm console encoding | Set `$env:PYTHONUTF8 = "1"` before running `uv run python -u notebooks/02_online_retail_ii_demand_forecast.py`. |
| Notebook result is stale | Inspect execution timestamp and CSV rows | Re-execute the matching notebook; do not relabel archive data as TimesFM 3.0 evidence. |

## Limits and escalation

TimesFM 3.0 weights are restricted to non-commercial, non-production use.
For a commercial or production requirement, stop before deployment and choose
a model and license appropriate to that use case. The project’s preflight and
notebook execution are reproducibility tools, not production readiness gates.
