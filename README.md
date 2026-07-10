# Demand Forecasting for Inventory Planning

**Tutorial + portfolio project:** forecast **aggregate unit demand** (not revenue) for inventory decisions, then compare two production-relevant approaches on the **same time-ordered holdout**:

1. **Classical / statistical path** — PyCaret 4.x time-series model survey (ranked by **MASE**) → **native reimplementation** of the winner (`statsmodels` / `pmdarima`) with residual diagnostics and prediction intervals  
2. **Foundation-model path** — Google **TimesFM 2.5** (200M params) **zero-shot** forecast: full history as inference context, **no gradient updates** on the local series  

Both tracks are fully implemented, executed, and scored with identical metrics (MAE, RMSE, MAPE, MASE). Notebooks ship with **real embedded plots and printed numbers** from a verified run on this machine—not placeholders.

| | |
|---|---|
| **Code license** | [MIT](LICENSE) |
| **Community** | [Contributing](CONTRIBUTING.md) · [Code of Conduct](CODE_OF_CONDUCT.md) · [Issues](https://github.com/pypi-ahmad/demand-forecasting-for-inventory-planning/issues/new/choose) |
| **Runtime** | Python **3.13.13**, [`uv`](https://github.com/astral-sh/uv), kernel `demand-forecast-project` |
| **Classical** | `pycaret[timeseries]==4.0.0a8`, statsmodels, pmdarima |
| **Foundation model** | `timesfm[torch]` → TimesFM 2.5 (`google/timesfm-2.5-200m-pytorch`, Apache-2.0 package) |
| **Hardware (this run)** | NVIDIA RTX 4060 Laptop (CUDA), ~30 GB system RAM |

---

## Who this README is for

| Audience | Jump to |
|----------|---------|
| **Portfolio / technical reviewers** | [Architecture & design decisions](#1-for-portfolio-evaluators--architecture-reproducibility-evidence) · [Real evidence](#real-results-this-run) · [Limitations](#honest-limitations) |
| **Hands-on operators** | [Installation](#2-for-hands-on-users--install-run-troubleshoot-extend) · [Run commands](#run-the-project) · [Troubleshooting](#troubleshooting) · [Extension points](#extending-the-local-workflow) |
| **Tutorial learners** | [Concepts](#3-for-tutorial-learners--concepts-and-implementation-flow) · [End-to-end flow](#implementation-flow-newbie--pro) · [Metrics glossary](#metrics-what-they-mean) |

---

## Problem statement

Inventory planning lives on **units over time**. A warehouse, DC, or retail planner needs:

- A **point forecast** of how many units will move in the next period(s)  
- An **uncertainty band** so service level can be traded against stockholding cost  
- A method that **does not leak the future** into training (strict time order)

This project does **not** build a full MRP / EOQ / multi-echelon optimizer. It builds the forecasting core those systems depend on, and ends each notebook with a **directional** inventory takeaway (point + band), not a formal safety-stock formula.

**Two datasets, same protocol:**

| Notebook | Dataset | Target |
|----------|---------|--------|
| [`notebooks/01_superstore_demand_forecast.ipynb`](notebooks/01_superstore_demand_forecast.ipynb) | Sample Superstore sales (line-level orders) | Σ `Quantity` per period |
| [`notebooks/02_online_retail_ii_demand_forecast.ipynb`](notebooks/02_online_retail_ii_demand_forecast.ipynb) | [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (id 502) | Σ `Quantity` per period |

Jupytext percent sources (`.py`) sit beside each notebook for script debugging.

---

# 1. For portfolio evaluators — architecture, reproducibility, evidence

## Design goals

1. **Evidence over folklore** — granularity (daily vs weekly), model winner, and “which approach won” are decided from **this run’s numbers**, not from generic Superstore/TimesFM blog posts.  
2. **Survey ≠ deliverable** — PyCaret narrows the field quickly; the real classical deliverable is a **native** fit with residuals and intervals.  
3. **Fair comparison** — same series, same holdout horizon `H`, same metric functions for classical and TimesFM.  
4. **Reproducible env** — single `uv` lockfile, pinned Python, registered Jupyter kernel, TimesFM system preflight script.  
5. **Honest failure modes** — e.g. `ucimlrepo` id 502 is often not importable; notebooks fall back to the **official UCI zip** and document that.

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                     Project root (flat layout)                    │
│  uv env · pyproject.toml · uv.lock · scripts/check_system.py      │
└─────────────────────────────────────────────────────────────────┘
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
   Notebook 01                                  Notebook 02
   Superstore CSV                               Online Retail II
   (GitHub raw)                                 (UCI 502 / zip)
          │                                           │
          └─────────────────────┬─────────────────────┘
                                ▼
              Clean → EDA → unit series → time split
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
     Part 1: PyCaret 4.x                   Part 2: TimesFM 2.5
     TimeSeriesExperiment                  from_pretrained + compile
     compare_models(sort=MASE)             forecast(horizon=H, inputs=[train])
              │                                   │
              ▼                                   │
     Native re-fit (statsmodels /                 │
     pmdarima) + ACF/PACF +                       │
     Ljung-Box + ~80% PI                          │
              │                                   │
              └─────────────────┬─────────────────┘
                                ▼
              MAE / RMSE / MAPE / MASE on holdout
              overlay plot + inventory narrative
```

### Critical architecture decisions (and why)

| Decision | Choice | Rationale | Tradeoff |
|----------|--------|-----------|----------|
| Target variable | **Units (`Quantity`)** | Inventory is unit-constrained; sales $ mixes price and promo | Ignores revenue mix and margin |
| Aggregation | **Chosen after daily vs weekly EDA** | Daily retail series are often sparse/noisy | Weekly loses day-of-week operational detail |
| Split | **Last H periods held out** | No random shuffle; no future leakage | Single cutpoint; sensitive to regime at end of series |
| Ranking metric | **MASE** (lower better) | Scale-free; PyCaret default for TS; comparable across models | Needs a sensible seasonal naive scale `m` |
| PyCaret role | **Survey only** | Fast multi-model CV; then leave the wrapper | Alpha API (4.0.0a8); not all models always succeed |
| Native re-fit | **Winner family in statsmodels/pmdarima** | Residual diagnostics + transparent intervals | Exotic winners may map to closest classical analogue |
| TimesFM mode | **Zero-shot, no fine-tune** | Demonstrates foundation prior without PEFT complexity | No domain adaptation; may underfit strong local seasonality |
| Survey seasonality | `seasonal_period=4` on weekly | Full annual `m=52` made auto_arima/ETS prohibitively slow | Survey seasonality ≠ full annual cycle |
| Survey model set | Classical shortlist (naive, ARIMA, ETS, theta, …) | Completes in minutes; natively re-implementable | Does not crown RF/GBR reduction models unless added |

### Reproducibility contract

| Artifact | Role |
|----------|------|
| `.python-version` → `3.13.13` | Exact interpreter pin for `uv` |
| `uv.lock` | Locked dependency graph |
| `pycaret[timeseries]==4.0.0a8` | Exact alpha pin (OOP API) |
| Kernel `demand-forecast-project` | Stable Jupyter target |
| `scripts/check_system.py` | TimesFM RAM/VRAM/disk preflight (from TimesFM agent skill) |
| Seeds | `session_id=42`, NumPy/Torch seed 42 |
| In-notebook download | Superstore URL; UCI zip with local cache under `data/` |
| Executed `.ipynb` | Committed with outputs (9 figures each on last full run) |

**This-run software stack (verified):**

```text
Python        3.13.13
pycaret       4.0.0a8
timesfm       2.0.2 (TimesFM_2p5_200M_torch present)
torch         2.13.0+cu130  |  CUDA: True
pandas        2.3.3
statsmodels   0.14.6
pmdarima      2.1.1
```

**Version-sensitive call chains actually used:**

```python
# PyCaret 4.x — OOP only (3.x bare setup()/compare_models() removed)
from pycaret.time_series import TimeSeriesExperiment
exp = TimeSeriesExperiment(fh=H, session_id=42, fold=2, seasonal_period=4, n_jobs=1)
exp.fit(y_train)  # univariate Series
result = exp.compare_models(sort="MASE", include=[...], turbo=True, n_select=1)
# result.best, result.leaderboard, result.ranked_ids

# TimesFM 2.5
import timesfm, torch
torch.set_float32_matmul_precision("high")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(timesfm.ForecastConfig(
    max_context=..., max_horizon=H, normalize_inputs=True,
    use_continuous_quantile_head=True, infer_is_positive=True,
    fix_quantile_crossing=True,
))
point, quantiles = model.forecast(horizon=H, inputs=[train_float32_1d])
```

### Evaluation protocol (identical on both notebooks)

1. Build one univariate series \(y_t\) = total units per period.  
2. Hold out final \(H=8\) periods as test; train = all prior points.  
3. PyCaret fits/CV on **train only**.  
4. Native model refit on **train only**; forecast \(H\) steps.  
5. TimesFM uses **train array as context** only; forecast \(H\) steps.  
6. Score both forecasts on the **same** test window with the **same** metric function.  
7. MASE scale uses seasonal naive on train with period `m` aligned to the survey cycle (weekly survey used `m=4` in this run’s implementation).

---

## Real results (this run)

Numbers below are taken from the **fully executed** notebooks on this machine (GPU TimesFM, live Superstore download, UCI zip for Retail II). Re-running may differ slightly due to solver paths or package patches; large qualitative conclusions should be re-checked.

### Notebook 1 — Superstore Sales

| Stage | Observed |
|-------|----------|
| Load | 9,994 × 21; encoding `utf-8-sig`; dates `dayfirst=True` (mixed D/M styles) |
| Clean keep rate | **9,994 / 9,994** |
| Calendar | 2014-01-03 → 2017-12-30 |
| Daily zero-order days | **15.2%** (221 / 1,458) |
| Daily vs weekly CV | **0.98** vs **0.57** |
| **Granularity decision** | **WEEKLY** |
| Weekly series | n=**209** (2014-01-05 → 2017-12-31) |
| ADF | stat=**−4.166**, p=**0.00075** → stationary at 5% |
| Train / test | n=**201** / **8** (test 2017-11-12 → 2017-12-31) |
| Mean demand train → test | **172.4 → 403.4** units/week (**year-end peak** in holdout) |
| PyCaret winner (expanding CV) | **`ets`** — CV MASE **0.7146**, CV RMSSE **0.5981**, CV MAE **55.35** |
| Native reimplementation | `statsmodels` ExponentialSmoothing (`seasonal=True`) |

**Holdout accuracy**

| Model | MAE | RMSE | MAPE (%) | MASE |
|-------|-----:|-----:|---------:|-----:|
| Native (ETS) | 104.84 | 119.45 | 26.61 | 1.3164 |
| **TimesFM 2.5 zero-shot** | **78.84** | **93.50** | **18.39** | **0.9899** |

**Holdout winner:** TimesFM 2.5 (all four metrics better).

**Uncertainty / inventory read (TimesFM):** mean point ≈ **346** units/week; lower band ≈ **243**; upper band ≈ **485** (width ≈ **242**). Actual holdout mean was **403**, inside the band but above the point forecast—consistent with a late-year spike the classical ETS under-shot.

---

### Notebook 2 — Online Retail II (UCI 502)

| Stage | Observed |
|-------|----------|
| Acquisition | `fetch_ucirepo(id=502)` → **DatasetNotFoundError (not available for import)**; fallback official zip `online+retail+ii.zip` (~45.6 MB), sheets 2009–2010 + 2010–2011 |
| Raw shape | **1,067,371** × 8 |
| Cleaning | Drop Invoice starting with `C` (19,494); non-positive qty/price; keep **1,041,670** |
| Calendar | 2009-12-01 → 2011-12-09 |
| Daily zero-order days | **18.3%** |
| Daily vs weekly CV | **0.87** vs **0.43** |
| **Granularity decision** | **WEEKLY** |
| Weekly series | n=**106** (2009-12-06 → 2011-12-11) |
| ADF | stat=**−5.053**, p=**1.74e-05** |
| Train / test | n=**98** / **8** |
| Mean demand train → test | **102,308 → 174,261** units/week |
| PyCaret winner (CV) | **`arima`** — CV MASE **1.4841**, CV MAE **63,175** |
| Native reimplementation | `pmdarima auto_arima` order **(1,0,0)**, seasonal **(0,0,0,4)** |
| Residuals | Ljung-Box p≈**0.35** → no strong residual autocorrelation |

**Holdout accuracy**

| Model | MAE | RMSE | MAPE (%) | MASE |
|-------|-----:|-----:|---------:|-----:|
| Native (ARIMA) | 69,829 | 76,436 | 38.52 | 1.8253 |
| **TimesFM 2.5 zero-shot** | **31,732** | **40,347** | **16.88** | **0.8295** |

**Holdout winner:** TimesFM 2.5 (large margin on MASE and MAPE).

**Uncertainty / inventory read (TimesFM):** mean point ≈ **146,495** units/week; band ≈ **95,877 – 203,930** (width ≈ **108,053**). Actual holdout mean **174,261** sits between point and upper band—again a higher-demand tail of the series.

### Cross-notebook interpretation (honest)

Both holdouts land on **elevated demand regimes** relative to train means. A short classical model (ETS / AR(1)-class) estimated on the quieter history **under-reacts**. TimesFM’s pretraining prior, given the full train context, tracked the step-up better **on these cuts**. That is **not** a universal claim that foundation models always beat ETS/ARIMA; it is what happened on these two series with H=8 and the survey settings above.

---

## Honest limitations

1. **PyCaret 4.0.0a8 is alpha** — API and model registry may change; pin the version.  
2. **Survey ≠ exhaustive AutoML** — classical shortlist only; several candidates can fail silently (`errors="ignore"`) and leave a thinner leaderboard.  
3. **Seasonal period in survey is shortened** (`m=4` weekly) for runtime; true annual weekly seasonality (`m=52`) was too expensive for auto_arima/ETS in practice.  
4. **H = 8 weeks** — short test window; different cut dates can reorder models.  
5. **Aggregate total units only** — no SKU hierarchy, no store panel, no price/promo covariates (TimesFM XReg exists but is unused).  
6. **Single train/test cut** — no rolling-origin full report in the notebook (PyCaret uses expanding CV internally for survey only).  
7. **Inventory close is qualitative** — not (Q,R) policy, not fill-rate optimization.  
8. **Data rights** — MIT covers **code**; Superstore sample, UCI CC BY 4.0, and TimesFM weights have separate terms.  
9. **No unsloth / no chat LLM** — TimesFM fine-tuning, if ever added, should follow Google’s PEFT/LoRA examples, not unsloth.  
10. **MAPE** uses an epsilon floor on near-zero actuals; still fragile if zeros appear in the holdout.

---

# 2. For hands-on users — install, run, troubleshoot, extend

## Prerequisites

- Linux (developed on Linux), `curl`, git optional  
- [uv](https://docs.astral.sh/uv/)  
- Network for first-time Superstore CSV, UCI zip, and Hugging Face TimesFM weights (~800 MB)  
- **Recommended:** NVIDIA GPU + CUDA (this run used RTX 4060). CPU works but is slower.  
- Disk: several GB free under `~/.cache/huggingface` and project `data/`

## Installation

```bash
cd "/path/to/Demand Forecasting for Inventory Planning"

# Interpreter
uv python install 3.13.13

# Env + deps (from lockfile)
uv sync

# Jupyter kernel used by both notebooks
uv run python -m ipykernel install --user --name demand-forecast-project

# TimesFM preflight — run before first model load
uv run python scripts/check_system.py
```

Expected preflight shape on a capable machine:

```text
VERDICT: ✅ System is ready for TimesFM 2.5 (200M) (GPU mode)
```

Confirm imports:

```bash
uv run python -c "
import pycaret, timesfm, torch
print(pycaret.__version__, hasattr(timesfm, 'TimesFM_2p5_200M_torch'),
      torch.__version__, torch.cuda.is_available())
"
```

## Run the project

### A. Fully re-execute notebooks (recommended for “real outputs”)

```bash
uv run jupyter nbconvert --to notebook --execute \
  notebooks/01_superstore_demand_forecast.ipynb \
  --output 01_superstore_demand_forecast.ipynb \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=demand-forecast-project

uv run jupyter nbconvert --to notebook --execute \
  notebooks/02_online_retail_ii_demand_forecast.ipynb \
  --output 02_online_retail_ii_demand_forecast.ipynb \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=demand-forecast-project
```

### B. Interactive

```bash
uv run jupyter lab
# Kernel: demand-forecast-project
# Open notebooks/01_*.ipynb or notebooks/02_*.ipynb
```

### C. Plain-script path (jupytext sources)

```bash
uv run python notebooks/01_superstore_demand_forecast.py
uv run python notebooks/02_online_retail_ii_demand_forecast.py
```

Scripts auto-detect non-IPython and use a non-blocking Matplotlib backend.

### D. Sync `.py` → `.ipynb` after editing sources

```bash
uv run jupytext --to ipynb notebooks/01_superstore_demand_forecast.py
uv run jupytext --to ipynb notebooks/02_online_retail_ii_demand_forecast.py
```

## Data paths (no manual download required)

| Dataset | Automatic path | Cache |
|---------|----------------|-------|
| Superstore | GitHub raw CSV inside notebook 01 | None (streamed each run) |
| Online Retail II | Try `ucimlrepo` id 502 → else UCI static zip | `data/online_retail_ii.zip` (gitignored) |
| TimesFM weights | Hugging Face `google/timesfm-2.5-200m-pytorch` | `~/.cache/huggingface/` |

Manual Retail II URL if you need it offline:

```text
https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
```

Place as `data/online_retail_ii.zip` and re-run notebook 02.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `requires-python` / pin conflict | System Python 3.14 vs project 3.13 | `uv python pin 3.13.13` and `uv sync` |
| `DatasetNotFoundError` id 502 | UCI disables Python import for 502 | Expected — notebook falls back to zip; ensure network or pre-seed `data/` |
| Superstore many NaT dates | Parsed as US M/D without `dayfirst` | Notebook uses `dayfirst=True`; do not switch to naive `to_datetime` |
| `compare_models` hangs | Seasonal `m=52` + auto_arima | Keep shortlist + `seasonal_period=4`; exclude `auto_arima` from survey if needed |
| Empty leaderboard | All candidates failed | Check traceback with `errors="raise"` temporarily; reduce `fh` / folds |
| TimesFM OOM | Low RAM/VRAM | Run `scripts/check_system.py`; lower `max_context`; use CPU if GPU VRAM &lt; 2 GB |
| `TimesFM_2p5_200M_torch` missing | Old `timesfm` | `uv add "timesfm[torch]"` (2.0.x ships 2.5 API) |
| CUDA False | CPU torch wheel | Install CUDA build of torch for your platform; project used `2.13.0+cu130` |
| No figures in executed nb | Headless `Agg` + close-all in IPython | Use `show_plot()` helper (IPython shows, script closes) |
| PyCaret `ImportError` for sktime | Missing extras | Install `pycaret[timeseries]==4.0.0a8` via `uv` |
| Permission / kernel not found | Kernel not registered | Re-run `ipykernel install --user --name demand-forecast-project` |

## Extending the local workflow

Practical extension points (smallest first):

1. **Change holdout length** — edit `H` in the target-series cell; keep PyCaret `fh` and TimesFM `horizon` equal.  
2. **Widen the survey** — add ids to `SURVEY_INCLUDE` (e.g. `auto_arima`, `stlf`) knowing runtime cost.  
3. **Annual seasonality** — set `seasonal_period=52` only if you accept long runtimes or exclude auto_arima.  
4. **SKU / category series** — filter before aggregation; loop TimesFM `inputs=[...]` as a batch.  
5. **Covariates** — TimesFM `forecast_with_covariates` + `timesfm[xreg]` (price, promo, holiday).  
6. **Rolling-origin evaluation** — wrap the metric block in a loop over cutpoints; store a leaderboard CSV under `data/results/`.  
7. **Fine-tuning** — follow `google-research/timesfm` PEFT/LoRA examples; **do not** force unsloth.  
8. **CLI wrapper** — thin `uv run python -m` entrypoint that runs one notebook script and writes `data/results/{dataset}_metrics.json` for CI.

Suggested metrics JSON shape for automation:

```json
{
  "dataset": "superstore",
  "grain": "weekly",
  "H": 8,
  "winner_pycaret": "ets",
  "native": {"MAE": 104.84, "RMSE": 119.45, "MAPE": 26.61, "MASE": 1.3164},
  "timesfm": {"MAE": 78.84, "RMSE": 93.50, "MAPE": 18.39, "MASE": 0.9899}
}
```

## Repository map

```text
.
├── LICENSE                          # MIT (code)
├── README.md
├── pyproject.toml / uv.lock
├── .python-version                  # 3.13.13
├── scripts/check_system.py          # TimesFM preflight
├── data/                            # UCI zip cache (gitignored)
└── notebooks/
    ├── 01_superstore_demand_forecast.py
    ├── 01_superstore_demand_forecast.ipynb   # executed
    ├── 02_online_retail_ii_demand_forecast.py
    └── 02_online_retail_ii_demand_forecast.ipynb
```

---

# 3. For tutorial learners — concepts and implementation flow

This section teaches **demand forecasting** ideas the notebooks implement. (There is no RAG stack here—retrieval/generation concepts do not apply.)

## Core concepts

### Demand vs sales dollars

- **Demand (units)** drives replenishment and capacity.  
- **Sales ($)** confounds price, discount, and mix.  
Both notebooks forecast **sum of Quantity**.

### Time order is a hard rule

Unlike churn classification, you **must not shuffle** rows. The model may only see \(y_1,\ldots,y_T\) when predicting \(y_{T+1},\ldots,y_{T+H}\). Random train/test splits leak the future and invent optimistic metrics.

### Aggregation grain

Raw data are **transactions**. Forecasting every raw timestamp is usually wrong for planning:

- **Daily** series often have many zero days (Superstore **15.2%**, Retail II **18.3%** in this run) and high CV.  
- **Weekly** smooths noise and matches many replenishment cadences.  

The notebooks **measure** sparsity/CV, then choose weekly when daily is clearly noisier—not because “weekly is always right.”

### Stationarity (ADF)

A rough check: does the series look like it has a unit root? Superstore weekly ADF p≈**0.00075**, Retail II p≈**1.7e-05** — both rejected unit root at 5% on the chosen series. That does **not** mean “no seasonality”; it means differencing may be less critical than for strong random-walk series. ARIMA/ETS can still use trend and seasonal components.

### Classical families (Part 1)

| Family | Intuition |
|--------|-----------|
| Naive / seasonal naive | “Tomorrow = today / same week last cycle” baseline |
| ETS / exponential smoothing | Level + trend + optional season; smooths noise |
| ARIMA | Linear dependence on lags + differencing |
| Theta | Decomposition-style baseline often strong on M-competitions |

PyCaret’s `compare_models(sort="MASE")` runs several of these under expanding-window CV and returns a **leaderboard**. The notebook then **re-implements the winner** outside PyCaret so you can inspect residuals:

- Residual time plot  
- ACF / PACF of residuals  
- Ljung-Box test (H0: no residual autocorrelation)

### Foundation models (Part 2) — zero-shot ≠ ignore history

TimesFM is pretrained on many series. On your data:

- **No backpropagation** on Superstore/Retail II  
- The train window is passed as **context** to `forecast(...)`  
- Output includes a **point** (median) and **quantile** path for bands  

Common misconception: “zero-shot means it doesn’t use my history.” False. It uses history at **inference**; it does not **update weights** from it.

### Metrics: what they mean

| Metric | Meaning | Notes |
|--------|---------|-------|
| **MAE** | Average absolute error in units | Same scale as demand |
| **RMSE** | Penalizes large misses more | Sensitive to spikes |
| **MAPE** | Relative error % | Unstable near zero actuals; notebooks use a floor |
| **MASE** | MAE / MAE of seasonal naive on train | **Primary ranking**; &lt; 1 beats seasonal naive scale |

Always compare models with the **same** definition and the **same** test indices.

### Inventory takeaway (directional)

Given point \(\hat y\) and band \([L, U]\):

- Ordering near \(\hat y\) optimizes for “average” demand.  
- Covering toward \(U\) raises service level and stock.  
- Covering toward \(L\) risks stockouts if the spike materializes.  

Notebooks print actual mean demand, preferred model’s mean point, and mean band edges from **this run**—use them as intuition, not as a compliance policy.

## Implementation flow (newbie → pro)

```text
1. Setup
   versions, seeds, CUDA check, assert PyCaret 4.x + TimesFM 2.5 symbols

2. Acquire data
   Superstore: HTTP CSV
   Retail II: ucimlrepo OR UCI zip → Excel both years

3. Clean
   Superstore: parse dates carefully; positive quantities
   Retail II: drop cancellations (Invoice ~ ^C), qty>0, price>0

4. EDA
   volume over time, missingness, daily vs weekly profiles, STL, ADF, ACF/PACF

5. Build y_t + split
   weekly (in both real runs) Σ Quantity
   train = all but last H; test = last H

6. Part 1 — survey
   TimeSeriesExperiment(fh=H).fit(train)
   compare_models(sort="MASE") → winner_id + leaderboard

7. Part 1 — native
   map winner → statsmodels/pmdarima
   residuals + PI + forecast plot

8. Part 2 — TimesFM
   from_pretrained → compile(ForecastConfig) → forecast
   quantile band plot

9. Compare
   metric table + overlay
   explain which won *on this series*

10. Inventory paragraph
    point + band → reorder/safety posture (qualitative)
```

### What “good” looks like when you re-run

- Notebook finishes with `Notebook 01 complete.` / `Notebook 02 complete.`  
- Leaderboard non-empty; winner named with MASE  
- Both metric dicts printed  
- Plots embedded if executed via Jupyter/nbconvert with IPython display  

---

## Data licenses and citation

| Asset | Terms |
|-------|--------|
| This repository’s code & notebooks | [MIT](LICENSE) |
| Online Retail II | CC BY 4.0 — Chen, D. (2019). UCI ML Repository. https://doi.org/10.24432/C5CG6D |
| TimesFM software | Apache-2.0 (Google Research); confirm weight terms on the Hugging Face model card |
| Superstore sample CSV | Third-party tutorial export — verify before commercial redistribution |

---

## References

- PyCaret: https://github.com/pycaret/pycaret  
- TimesFM: https://github.com/google-research/timesfm  
- TimesFM paper: *A decoder-only foundation model for time-series forecasting* (ICML 2024), https://arxiv.org/abs/2310.10688  
- UCI Online Retail II: https://archive.ics.uci.edu/dataset/502/online+retail+ii  
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* — MASE and time series CV background  

---

## Quick status checklist

| Check | Status on last full verification |
|-------|----------------------------------|
| `uv sync` env Python 3.13.13 | OK |
| Kernel `demand-forecast-project` | OK |
| TimesFM system check | READY (GPU) |
| NB01 executed, 9 figures, 0 errors | OK |
| NB02 executed, 9 figures, 0 errors | OK |
| README metrics match notebook streams | OK (tables above) |
| MIT `LICENSE` present | OK |

For a clean re-verification after changes: re-run both `nbconvert --execute` commands and diff the holdout metric dicts against the tables in [Real results](#real-results-this-run).
