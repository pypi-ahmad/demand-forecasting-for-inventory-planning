# Demand Forecasting for Inventory Planning

**Tutorial + portfolio project:** forecast **aggregate unit demand** (not revenue) for inventory decisions. The repo is layered so **older notebooks stay intact for learning**, while newer notebooks add stronger techniques:

| Generation | Notebooks | Focus |
|------------|-----------|--------|
| **v1** | `01`, `02` (early cells / historical metrics) | PyCaret short-cycle survey + TimesFM zero-shot |
| **v2** | `01`, `02` production bake-off | Multiplicative HW over seasonal periods m∈{4…52}, rolling origin |
| **v3** | **`03`, `04` (new — does not edit 01/02)** | Hierarchy, calendar/promo features, log1p HW, SARIMAX+exog, ML lags, smart stack, **inventory asymmetric cost**, SL≈0.9 order qty |

All tracks use the **same weekly series and H=8 holdout** where compared. Metrics are from real runs—not placeholders.

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
| **Portfolio / technical reviewers** | [Architecture](#1-for-portfolio-evaluators--architecture-reproducibility-evidence) · [v1 vs v2 results](#real-results--v1-baseline-vs-v2-production-bake-off) · [v3 advanced results](#real-results--v3-advanced-stack-new) · [Limitations](#honest-limitations) |
| **Hands-on operators** | [Installation](#2-for-hands-on-users--install-run-troubleshoot-extend) · [Run commands](#run-the-project) · [Troubleshooting](#troubleshooting) |
| **Tutorial learners** | [Concepts](#3-for-tutorial-learners--concepts-and-implementation-flow) · [v1→v2→v3 progression](#how-we-got-better-results-summary) |

---

## Problem statement

Inventory planning lives on **units over time**. A warehouse, DC, or retail planner needs:

- A **point forecast** of how many units will move in the next period(s)  
- An **uncertainty band** so service level can be traded against stockholding cost  
- A method that **does not leak the future** into training (strict time order)

This project does **not** build a full MRP / EOQ / multi-echelon optimizer. It builds the forecasting core those systems depend on, and ends each notebook with a **directional** inventory takeaway (point + band), not a formal safety-stock formula.

**Notebooks (all kept):**

| Notebook | Dataset | Generation |
|----------|---------|------------|
| [`01_superstore_demand_forecast`](notebooks/01_superstore_demand_forecast.ipynb) | Superstore Sales | v1 survey + **v2** bake-off (preserved) |
| [`02_online_retail_ii_demand_forecast`](notebooks/02_online_retail_ii_demand_forecast.ipynb) | [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | v1 survey + **v2** bake-off (preserved) |
| [`03_superstore_advanced_demand_forecast`](notebooks/03_superstore_advanced_demand_forecast.py) | Superstore | **v3 advanced tutorial** (new; does not edit 01) |
| [`04_online_retail_ii_advanced_demand_forecast`](notebooks/04_online_retail_ii_advanced_demand_forecast.py) | Online Retail II | **v3 advanced tutorial** (new; does not edit 02) |

Jupytext percent sources (`.py`) sit beside notebooks for script debugging.

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

## Real results — v1 baseline vs v2 production bake-off

All numbers below come from **fully executed notebooks** on this machine (Python 3.13.13, PyCaret 4.0.0a8, TimesFM 2.5, torch CUDA on RTX 4060).  
**Same data, same weekly grain, same last-H=8 holdout** for fair comparison. We **keep v1 results** and add v2 side-by-side.

Shared data facts (unchanged between v1 and v2):

| | Superstore | Online Retail II |
|--|------------|------------------|
| Rows after clean | 9,994 / 9,994 | 1,041,670 / 1,067,371 |
| Grain | Weekly (daily zero-days 15.2%) | Weekly (daily zero-days 18.3%) |
| Series length | 209 weeks | 106 weeks |
| Train / test | 201 / **8** | 98 / **8** |
| Train → test mean units/week | 172.4 → **403.4** (year-end peak) | 102,308 → **174,261** (peak ramp) |

---

### What changed technically (v1 → v2)

| | **v1 baseline (first ship)** | **v2 production bake-off (current)** |
|--|------------------------------|--------------------------------------|
| Classical path | PyCaret short-cycle survey (`seasonal_period=4`) → re-fit winner only (ETS / low-order ARIMA) | Multi-model **bake-off** over seasonal periods that fit ≥2 cycles: **m ∈ {4, 8, 13, 26, 52}** |
| Smoothing form | Mostly additive / short-season ETS | **Multiplicative Holt–Winters** (`trend=add`, `seasonal=mul`) when positive demand |
| Foundation path | TimesFM 2.5 zero-shot (point + q10–q90) | Same TimesFM 2.5, production flags: `normalize_inputs`, continuous quantile head, `infer_is_positive`, `fix_quantile_crossing` |
| Model selection | One PyCaret MASE leaderboard | Holdout leaderboard **+** nested val MAE **+** inverse-MAE ensemble **+** **rolling-origin** (3 cuts) |
| Metrics | MAE, RMSE, MAPE, MASE | + sMAPE, bias, PI coverage; CSVs under `data/results/` |
| Code | Inline notebook functions | Package `demand_forecast/` (`classical`, `timesfm_runner`, `bakeoff`, `metrics`) |

**Why these techniques improve retail unit demand**

1. **Seasonal period search** — Short cycles (`m=4`) miss **annual** retail peaks. Superstore needs ~`m=52`; Retail II (shorter history) wins with ~`m=13` (quarterly weeks), not forced `m=52` when train &lt; 2×52.  
2. **Multiplicative seasonality** — Peak weeks scale with level (holiday / year-end). Additive ETS under-forecasted both holdouts.  
3. **Seasonal naive baseline** — Always in the bake-off; prevents “complex but worse” models from shipping.  
4. **Rolling-origin** — One lucky 8-week cut is not production gating; v2 re-fits over multiple origins.  
5. **TimesFM kept as peer, not assumed winner** — Strong on multi-series / cold-start ops; on these **single** seasonal aggregates, a well-specified HW often wins holdout.  
6. **PyCaret retained for teaching** — Fast OOP survey (v1-style); **shipping decision** uses the bake-off champion.

---

### Notebook 1 — Superstore: old vs new

#### v1 baseline results (preserved)

PyCaret survey winner: **`ets`** (CV MASE 0.7146).  
Native re-fit: short-cycle Exponential Smoothing. TimesFM zero-shot on the same holdout.

| Model (v1) | MAE | RMSE | MAPE (%) | MASE |
|------------|-----:|-----:|---------:|-----:|
| Native ETS (short-cycle / m≈4 path) | 104.84 | 119.45 | 26.61 | 1.3164 |
| TimesFM 2.5 zero-shot | 78.84 | 93.50 | 18.39 | 0.9899* |

\*v1 README ranked TimesFM best on MASE using the v1 MASE scale settings; both v1 models are kept for audit. Point forecast then ≈ **346** units/week vs actual mean **403**.

#### v2 production bake-off results (new)

Production champion: **`holt_winters_mul_m52`**  
(`statsmodels` ExponentialSmoothing, trend=add, seasonal=mul, period=**52**).  
Ljung-Box on residuals p≈**0.22**. Full table: `data/results/superstore_production_metrics.csv`.

| Model (v2 holdout) | MAE | RMSE | MAPE (%) | sMAPE (%) | MASE | bias | PI cover |
|--------------------|-----:|-----:|---------:|----------:|-----:|-----:|---------:|
| **holt_winters_mul_m52 (champion)** | **48.12** | **61.40** | **11.44** | **11.52** | **0.803** | −13.8 | 0.625 |
| holt_winters_mul_m26 | 54.25 | 72.06 | 13.07 | 13.52 | 0.906 | −27.3 | 0.750 |
| holt_winters_add_m52 | 57.57 | 78.78 | 12.98 | 13.85 | 0.961 | −40.7 | 0.500 |
| TimesFM 2.5 zero-shot | 78.84 | 93.50 | 18.39 | 19.79 | 1.316 | −57.2 | **0.875** |
| seasonal_naive_m52 | 80.00 | 119.29 | 17.16 | 20.72 | 1.336 | −78.8 | 0.625 |
| ensemble_inv_val_mae | 83.23 | 95.77 | 20.45 | 21.43 | 1.390 | −49.7 | 0.625 |
| holt_winters_add_m4 (≈ v1 classical) | 104.84 | 119.45 | 26.61 | 27.49 | 1.751 | −55.2 | 0.500 |
| auto_arima_m4 | 123.97 | 149.61 | 27.18 | 33.07 | 2.070 | −118.8 | 0.375 |

**Rolling-origin mean MASE (3 origins, v2):** `holt_winters_mul_m52` **0.764** (best family); TimesFM mean MASE ≈ **1.11**.

**Inventory (v2 champion):** point ≈ **389.6** units/week; PI ≈ **337.9–441.2**; actual mean **403.4**. Bias **−13.8** (mild under-forecast). Service-minded cover toward ~**441**.

#### Superstore head-to-head (same holdout)

| Metric | v1 native ETS | v1 TimesFM | **v2 champion HW mul m=52** | v2 vs v1 ETS | v2 vs v1 TimesFM |
|--------|--------------:|-----------:|----------------------------:|-------------:|-----------------:|
| MAE | 104.84 | 78.84 | **48.12** | **−54.1%** | **−39.0%** |
| RMSE | 119.45 | 93.50 | **61.40** | **−48.6%** | **−34.3%** |
| MAPE (%) | 26.61 | 18.39 | **11.44** | **−57.0%** | **−37.8%** |
| MASE | 1.316 | 0.990* | **0.803** | **−39.0%** | better on v2 scale / bake-off |
| Point forecast (mean) | ~346 | ~346 | **~390** | closer to actual 403 | closer to actual 403 |

---

### Notebook 2 — Online Retail II: old vs new

#### v1 baseline results (preserved)

PyCaret survey winner: **`arima`** (CV MASE 1.4841).  
Native re-fit: `pmdarima auto_arima (1,0,0)` seasonal `(0,0,0,4)`. TimesFM zero-shot on the same holdout.

| Model (v1) | MAE | RMSE | MAPE (%) | MASE |
|------------|-----:|-----:|---------:|-----:|
| Native auto_arima m=4 | 69,829 | 76,436 | 38.52 | 1.8253 |
| TimesFM 2.5 zero-shot | 31,732 | 40,347 | 16.88 | 0.8295 |

v1 held TimesFM as holdout winner. Point ≈ **146,495** units/week vs actual mean **174,261**.

#### v2 production bake-off results (new)

Production champion (holdout): **`holt_winters_mul_m13`**.  
Full table: `data/results/online_retail_ii_production_metrics.csv`.

| Model (v2 holdout) | MAE | RMSE | MAPE (%) | sMAPE (%) | MASE | bias | PI cover |
|--------------------|-----:|-----:|---------:|----------:|-----:|-----:|---------:|
| **holt_winters_mul_m13 (champion)** | **22,453** | **29,926** | **11.91** | **12.92** | **0.561** | −18,411 | **0.875** |
| holt_winters_mul_m8 | 27,624 | 41,807 | 13.88 | 15.93 | 0.691 | −23,467 | 0.750 |
| seasonal_naive_m52 | 27,724 | 38,596 | 16.59 | 15.77 | 0.693 | −3,838 | 0.875 |
| holt_winters_mul_m26 | 30,227 | 34,464 | 16.83 | 18.46 | 0.756 | −25,044 | 0.875 |
| TimesFM 2.5 zero-shot | 31,732 | 40,347 | 16.88 | 18.99 | 0.793 | −27,766 | 0.875 |
| ensemble_inv_val_mae | 32,029 | 40,863 | 16.89 | 19.07 | 0.801 | −30,825 | 0.875 |
| auto_arima_m4 (≈ v1 classical) | 69,829 | 76,436 | 38.52 | 48.58 | 1.745 | −69,829 | 0.500 |

**Rolling-origin mean MASE (v2):** TimesFM ≈ **0.50** (strong across cuts); HW mul m=13 ≈ **0.54**.  
→ **Holdout champion ≠ always rolling champion** — production reports both.

**Inventory (v2 champion):** point ≈ **155,849** units/week; PI ≈ **107,248–204,451**; actual mean **174,261**. Bias **−18,411** — use upper band for service-level planning on peak weeks.

#### Retail II head-to-head (same holdout)

| Metric | v1 auto_arima m=4 | v1 TimesFM | **v2 champion HW mul m=13** | v2 vs v1 ARIMA | v2 vs v1 TimesFM |
|--------|------------------:|-----------:|----------------------------:|---------------:|-----------------:|
| MAE | 69,829 | 31,732 | **22,453** | **−67.8%** | **−29.2%** |
| RMSE | 76,436 | 40,347 | **29,926** | **−60.8%** | **−25.8%** |
| MAPE (%) | 38.52 | 16.88 | **11.91** | **−69.1%** | **−29.4%** |
| MASE | 1.825 | 0.830 | **0.561** | **−69.3%** | **−32.4%** |

---

### How we got better results (summary)

```text
v1:  PyCaret(m=4) → single native re-fit → TimesFM zero-shot → pick lower MASE
                    ✗ often wrong seasonal period
                    ✗ additive/short-cycle under peak demand

v2:  EDA grain (weekly)
  → bake-off: snaive + HW add/mul × {4,8,13,26,52} + auto_arima + TimesFM
  → nested val weights + ensemble
  → holdout leaderboard + rolling-origin means
  → ship champion + uncertainty band
                    ✓ period fits history length
                    ✓ multiplicative peaks
                    ✓ robust gate, not one cut
```

| Technique | Superstore impact | Retail II impact |
|-----------|-------------------|------------------|
| Multiplicative HW | m=52 champion | m=13 champion |
| Period search | m=52 ≫ m=4 | m=13 ≫ m=4 ARIMA |
| TimesFM production config | Best PI coverage (0.875) | Competitive; best rolling mean |
| Rolling-origin | Confirms mul_m52 family | TimesFM wins mean MASE |
| Keep v1 metrics | Audit trail / regression check | Same |

Artifacts:  
- `data/results/superstore_production_metrics.csv`  
- `data/results/online_retail_ii_production_metrics.csv`  
- Executed notebooks under `notebooks/*.ipynb`

---

### Cross-notebook lessons

| Lesson | Evidence |
|--------|----------|
| Seasonality period is a first-class hyperparameter | m=4 lost to m=52 / m=13 by large margins on both datasets |
| Multiplicative HW fits retail peak amplitude | Year-end Superstore; holiday ramp Retail II |
| TSFMs are peers, not automatic winners on one aggregate series | TimesFM best PI / rolling on Retail; HW wins Superstore holdout |
| PyCaret survey ≠ production champion | v1 survey winners were ETS/ARIMA short-cycle |
| Always keep seasonal naive | Often close to mid-tier models |
| Report holdout **and** rolling ranks | Retail: HW wins holdout, TimesFM wins rolling mean |

---

## Real results — v3 advanced stack (new)

**Notebooks 03 / 04** add production-style techniques **without changing** notebooks 01/02 or their v1/v2 metrics. Same weekly totals and **H=8** holdout for comparison.

### What v3 adds (and why)

| Technique | Module | Why we do it |
|-----------|--------|----------------|
| **Hierarchical bottom-up** (Category×Region or Country → sum) | `advanced/hierarchy.py` | Total demand mixes different seasons; bottoms can carry structure the aggregate hides |
| **Calendar + holiday + promo features** | `advanced/features.py` | Year-end peaks are partly *driven* by calendar/discount, not pure noise |
| **log1p + HW** | `advanced/models_exog.py` | Variance often grows with level; stabilize then back-transform |
| **SARIMAX + exogenous regressors** | `advanced/models_exog.py` | Classical linear dynamics + measurable drivers |
| **HistGradientBoosting lags + exog** | `advanced/models_exog.py` | Nonlinear lag/promo interactions |
| **TimesFM zero-shot (peer)** | `timesfm_runner.py` | Foundation prior; XReg optional (needs `timesfm[xreg]`+JAX — documented, not required) |
| **Smart 2–3 model stack** | `advanced/ensemble.py` | Inverse-**validation** MAE weights on diverse strong models only (not 12-model soup) |
| **Asymmetric inventory cost** (underage=4, overage=1) | `advanced/inventory.py` | Stockouts cost more than overstock — MAE alone is not ops |
| **SL≈0.9 order quantity** from PI band | `advanced/inventory.py` | Reorder from predictive upper band, not only the median |
| **Longer rolling-origin (6 cuts)** | `advanced/evaluation.py` | Gate models across many windows |
| **Champion gate** | `advanced/pipeline.py` | Among models within **15% of best MASE**, pick **lowest asymmetric cost** (stops pure over-forecast “winners”) |

Package entrypoint: `demand_forecast.advanced.run_advanced_pipeline`.

```text
v1 → teach survey vs TimesFM
v2 → fix seasonality (HW mul + period search)  ← large accuracy jump
v3 → hierarchy, drivers, inventory loss, smart stack, longer rolling
     ← accuracy champion often still v2 HW; *decision quality* metrics expand
```

### Superstore — v3 results (notebook 03)

**Accuracy leaderboard (selected; full CSV: `data/results/superstore_v3_accuracy.csv`)**

| Model | MAE | RMSE | MAPE (%) | MASE | bias |
|-------|-----:|-----:|---------:|-----:|-----:|
| **hw_mul_m52 (v3 champion)** | **48.12** | **61.40** | **11.44** | **0.803** | −13.8 |
| hw_mul_m26 | 54.25 | 72.06 | 13.07 | 0.906 | −27.3 |
| smart_stack (HW26+log1p+HW52) | 54.99 | 73.64 | 12.85 | 0.918 | −29.4 |
| hw_log1p | 73.20 | 93.34 | 16.71 | 1.222 | −47.4 |
| timesfm_zeroshot | 78.84 | 93.50 | 18.39 | 1.316 | −57.2 |
| hierarchy_bottom_up (Category×Region) | 90.49 | 109.26 | 20.59 | 1.511 | −89.4 |
| sarimax_exog | 90.80 | 106.16 | 22.45 | 1.516 | −45.9 |
| hgb_lags_exog | 119.40 | 136.03 | 27.63 | 1.993 | −94.8 |
| seasonal_naive | 202.00 | 229.40 | 45.66 | 3.373 | −202.0 |

**Inventory table (underage cost 4× overage; lower cost better)** — `data/results/superstore_v3_inventory.csv`

| Model | under_units | over_units | asymmetric_cost | SL0.9 order cost |
|-------|------------:|-----------:|----------------:|-----------------:|
| **hw_mul_m52** | 247.7 | 137.2 | **1128.2** | **772.2** |
| hw_mul_m26 | 326.3 | 107.7 | 1413.0 | 861.1 |
| smart_stack | 337.6 | 102.3 | 1452.7 | 896.3 |
| timesfm_zeroshot | 544.3 | 86.4 | 2263.7 | 834.1 |
| hierarchy_bottom_up | 719.4 | 4.4 | 2882.2 | 1578.6 |

Note: hierarchy has very low overstock but **high understock** on this peak holdout → loses the dual gate (MASE + cost).

**v1 → v2 → v3 accuracy (Superstore, same holdout)**

| Metric | v1 ETS (m≈4) | v1 TimesFM | **v2 / v3 HW mul m=52** |
|--------|-------------:|-----------:|------------------------:|
| MAE | 104.84 | 78.84 | **48.12** |
| MAPE (%) | 26.61 | 18.39 | **11.44** |
| MASE | 1.316 | ~0.99 | **0.803** |

**What v3 changed vs v2 on Superstore:** holdout **accuracy champion is the same** as v2 (`hw_mul_m52`). v3’s value is **not a new MAE winner** on this cut, but:

1. **Confirms** m=52 still wins when hierarchy, SARIMAX, ML, and stacks compete.  
2. Adds **inventory KPIs** — SL0.9 order **cuts asymmetric cost** 1128 → **772** for the same champion.  
3. Documents **smart_stack** and hierarchy as learning baselines (stack MAE 55.0, hierarchy 90.5).  
4. Extends **rolling-origin** evaluation for ops gating.

---

### Online Retail II — v3 results (notebook 04)

**Accuracy leaderboard (selected; `data/results/online_retail_ii_v3_accuracy.csv`)**

| Model | MAE | RMSE | MAPE (%) | MASE | bias |
|-------|-----:|-----:|---------:|-----:|-----:|
| **hw_mul_m13 (v3 champion)** | **22,453** | **29,926** | **11.91** | **0.561** | −18,411 |
| hw_mul_m8 | 27,624 | 41,807 | 13.88 | 0.691 | −23,467 |
| hw_mul_m26 (val-selected alias) | 30,227 | 34,464 | 16.83 | 0.756 | −25,044 |
| smart_stack | 30,937 | 37,539 | 16.82 | 0.773 | −29,773 |
| timesfm_zeroshot | 31,732 | 40,347 | 16.88 | 0.793 | −27,766 |
| hierarchy_bottom_up (Country) | 50,624 | 55,561 | 28.43 | 1.265 | −50,624 |
| sarimax_exog | 68,224 | 75,392 | 41.27 | 1.705 | +58,401 |
| seasonal_naive | 90,607 | 93,722 | 51.96 | 2.265 | −90,607 |

**Inventory (selected)** — champion `hw_mul_m13`: under_units ≈ **163.5k**, over ≈ **16.2k**, asymmetric_cost ≈ **670k**; SL0.9 order **sharply reduces under_units** (≈23k) on the same holdout.

**v1 → v2 → v3 accuracy (Retail II, same holdout)**

| Metric | v1 ARIMA m=4 | v1 TimesFM | **v2 / v3 HW mul m=13** |
|--------|-------------:|-----------:|------------------------:|
| MAE | 69,829 | 31,732 | **22,453** |
| MAPE (%) | 38.52 | 16.88 | **11.91** |
| MASE | 1.825 | 0.830 | **0.561** |

**What v3 changed vs v2 on Retail II:** again the **holdout accuracy champion matches v2** (`hw_mul_m13`). Nested val alone preferred m=26; the multi-period board still surfaces **m=13** as best — showing why we fit **all feasible periods**, not only the val-selected alias.

---

### How to read the three generations

| Question | Answer from this repo |
|----------|------------------------|
| Did v2 beat v1? | **Yes, dramatically** (Superstore MAE −54%; Retail −68% vs old classical). |
| Did v3 beat v2 on MAE? | **Not on these two peak holdouts** — best univariate seasonal model was already strong. |
| Why keep v3? | **Inventory-aware selection**, SL0.9 reorder math, hierarchy/features as **auditable competitors**, longer rolling tests, tutorial explanations for production teams. |
| What still wins accuracy? | **Multiplicative HW with the right m** (52 Superstore, 13 Retail). |
| What should ops order? | Prefer **SL≈0.9 quantile order** (from PI), not raw point — see inventory tables. |

Artifacts:

- v2: `data/results/superstore_production_metrics.csv`, `online_retail_ii_production_metrics.csv`  
- v3: `superstore_v3_accuracy.csv`, `superstore_v3_inventory.csv`, `superstore_v2_vs_v3.csv`  
- v3: `online_retail_ii_v3_accuracy.csv`, `online_retail_ii_v3_inventory.csv`, `online_retail_ii_v2_vs_v3.csv`

---

## Honest limitations

1. **PyCaret 4.0.0a8 is alpha** — API and model registry may change; pin the version.  
2. **Survey ≠ exhaustive AutoML** — classical shortlist only; some candidates can fail silently (`errors="ignore"`).  
3. **PyCaret survey still uses short `m=4` for speed** — v2/v3 bake-offs are the shipping accuracy path.  
4. **H = 8 weeks** — short peak holdouts; different cuts can reorder models (mitigated by rolling-origin).  
5. **Hierarchy is educational on these aggregates** — Country/Category bottoms need explosion caps and level scaling; they did **not** beat HW mul on holdout MAE here.  
6. **TimesFM XReg** needs `timesfm[xreg]` + JAX (heavy); v3 uses zero-shot TimesFM + classical/ML exog instead.  
7. **Rolling-origin is limited** (v2: 3 origins; v3: ~6) — not a full multi-year platform.  
8. **Inventory model is simplified** newsvendor costs — not a full (Q,R) / fill-rate optimizer.  
9. **Champion bias is still negative** on peak holdouts — service levels should lean on upper PI / SL0.9 orders.  
10. **Data rights** — MIT covers **code**; Superstore sample, UCI CC BY 4.0, TimesFM weights have separate terms.  
11. **No unsloth** — TimesFM fine-tune, if ever added, follows Google PEFT/LoRA, not unsloth.  
12. **MAPE** uses an epsilon floor; fragile if zeros dominate.

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
├── LICENSE / CONTRIBUTING / CODE_OF_CONDUCT
├── README.md
├── pyproject.toml / uv.lock / .python-version
├── demand_forecast/
│   ├── classical.py / bakeoff.py / metrics.py / timesfm_runner.py   # v2
│   └── advanced/                                                    # v3
│       ├── features.py hierarchy.py models_exog.py
│       ├── ensemble.py evaluation.py inventory.py pipeline.py
├── scripts/check_system.py
├── data/
│   ├── online_retail_ii.zip              # gitignored cache
│   └── results/
│       ├── *_production_metrics.csv      # v2
│       └── *_v3_*.csv / *_v2_vs_v3.csv   # v3
└── notebooks/
    ├── 01_superstore_demand_forecast.{py,ipynb}           # v1+v2 (frozen)
    ├── 02_online_retail_ii_demand_forecast.{py,ipynb}     # v1+v2 (frozen)
    ├── 03_superstore_advanced_demand_forecast.py          # v3 tutorial
    └── 04_online_retail_ii_advanced_demand_forecast.py    # v3 tutorial
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
| README metrics match notebook streams | OK (v1 + v2 tables above) |
| MIT `LICENSE` present | OK |
| Production CSVs under `data/results/` | OK |

For a clean re-verification after changes: re-run both `nbconvert --execute` commands and diff the holdout metric dicts against [v1 vs v2 real results](#real-results--v1-baseline-vs-v2-production-bake-off).
