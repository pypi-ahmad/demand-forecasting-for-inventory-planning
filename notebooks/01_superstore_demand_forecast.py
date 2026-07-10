# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: demand-forecast-project
#     language: python
#     name: demand-forecast-project
# ---

# %% [markdown]
# # Demand Forecasting for Inventory Planning — Superstore Sales
#
# **Goal.** Build a unit-demand forecast that can inform inventory decisions
# (stockouts vs. overstock), then compare two very different approaches on the
# **same held-out period**:
#
# 1. **Part 1 — classical / ML survey with PyCaret 4.x**, then a *native*
#    reimplementation of the winner (statsmodels / pmdarima), with residual
#    diagnostics and prediction intervals.
# 2. **Part 2 — Google TimesFM 2.5** zero-shot foundation-model forecast
#    (no gradient updates on this series; history is inference-time context).
#
# **Business framing.** Inventory planning lives or dies on *units* over time,
# not just revenue. Under-forecast → stockouts and lost sales; over-forecast →
# excess inventory, cash tied up, and markdown risk. A useful forecast is
# therefore (a) accurate on a time-ordered holdout and (b) honest about
# uncertainty so you can set a safety buffer.
#
# This notebook teaches the pipeline **newbie → pro**: raw transactions →
# clean series → evidence-based aggregation → model survey → proper classical
# fit → foundation-model zero-shot → fair metrics → inventory takeaway.

# %% [markdown]
# ## 1. Setup
#
# We pin a single `uv`-managed kernel (`demand-forecast-project`) for both
# notebooks in this project. PyCaret **4.0** is OOP-only
# (`TimeSeriesExperiment`); TimesFM **2.5** is loaded via `timesfm[torch]`.

# %%
from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Project root on path for production package
_ROOT = Path.cwd() if (Path.cwd() / 'pyproject.toml').exists() else Path.cwd().parent
if str(_ROOT) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import torch

try:
    from IPython import get_ipython
    from IPython.display import display

    IN_IPYTHON = get_ipython() is not None
except ImportError:  # plain script execution
    IN_IPYTHON = False

    def display(obj):
        print(obj)

import matplotlib

if not IN_IPYTHON:
    matplotlib.use("Agg")  # headless script runs only
import matplotlib.pyplot as plt


def show_plot() -> None:
    """Show in Jupyter (embed outputs); close in plain-script runs."""
    plt.tight_layout()
    if IN_IPYTHON:
        plt.show()
    else:
        plt.show(block=False)
        plt.close("all")
from pmdarima import auto_arima
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (11, 4)
plt.rcParams["axes.titlesize"] = 12

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Python        : {sys.version.split()[0]}")
print(f"pandas        : {pd.__version__}")
print(f"numpy         : {np.__version__}")
print(f"torch         : {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU           : {torch.cuda.get_device_name(0)}")

import pycaret
import timesfm

print(f"pycaret       : {pycaret.__version__}")
print(f"timesfm 2.5   : TimesFM_2p5_200M_torch={hasattr(timesfm, 'TimesFM_2p5_200M_torch')}")
print(f"Kernel target : demand-forecast-project")
assert pycaret.__version__.startswith("4."), "Expected PyCaret 4.x OOP API"
assert hasattr(timesfm, "TimesFM_2p5_200M_torch"), "TimesFM 2.5 torch class missing"

# %% [markdown]
# ## 2. Data acquisition — Superstore Sales
#
# The classic Sample Superstore export: one row per line item with order dates,
# categories, and **Quantity** (units). We load it live from GitHub so the
# notebook is self-contained. Encoding note: the raw file ships with a UTF-8
# BOM, so `utf-8-sig` is the right first try (latin-1 is a common fallback for
# older Superstore copies).

# %%
PRIMARY_URL = (
    "https://raw.githubusercontent.com/yajasarora/"
    "Superstore-Sales-Analysis-with-Tableau/master/"
    "Superstore%20sales%20dataset.csv"
)
FALLBACK_URL = (
    "https://github.com/yajasarora/Superstore-Sales-Analysis-with-Tableau/"
    "raw/refs/heads/master/Superstore%20sales%20dataset.csv"
)

def load_superstore() -> pd.DataFrame:
    for url in (PRIMARY_URL, FALLBACK_URL):
        for enc in ("utf-8-sig", "latin-1"):
            try:
                frame = pd.read_csv(url, encoding=enc)
                print(f"Loaded via encoding={enc!r}\n  url={url}")
                return frame
            except Exception as exc:  # noqa: BLE001 — try next candidate
                print(f"Failed ({enc}): {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load Superstore CSV from either URL")

df_raw = load_superstore()
print(f"Shape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")
display(df_raw.head(3))
display(df_raw.dtypes.to_frame("dtype"))

# %% [markdown]
# ## 3. Exploratory data analysis
#
# Inventory forecasting needs a clean **unit demand** timeline. We inspect
# missingness, quantity outliers, and the temporal span before choosing an
# aggregation grain.

# %%
# Normalize column names that may have stray spaces
df = df_raw.copy()
df.columns = [c.strip() for c in df.columns]

required = {"Order Date", "Quantity"}
missing_cols = required - set(df.columns)
if missing_cols:
    raise KeyError(f"Missing expected columns: {missing_cols}")

# This export mixes D/M/Y styles; dayfirst=True + coerce keeps all valid calendar dates.
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce", dayfirst=True)
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

print("=== Missingness ===")
print(df[["Order Date", "Quantity"]].isna().sum())
print(f"\nDate range: {df['Order Date'].min()} → {df['Order Date'].max()}")
print(f"Rows: {len(df):,}")
print(f"Duplicate Row ID: {df['Row ID'].duplicated().sum() if 'Row ID' in df.columns else 'n/a'}")

print("\n=== Quantity distribution ===")
print(df["Quantity"].describe(percentiles=[0.01, 0.5, 0.99]))
print(f"Non-positive Quantity rows: {(df['Quantity'] <= 0).sum()}")
print(f"Null Order Date rows: {df['Order Date'].isna().sum()}")

# Drop unusable rows for demand aggregation
n_before = len(df)
df = df.dropna(subset=["Order Date", "Quantity"])
df = df[df["Quantity"] > 0].copy()
print(f"\nKept {len(df):,} / {n_before:,} rows after date/quantity cleaning")

# Daily order counts (sparsity check)
daily_orders = df.groupby(df["Order Date"].dt.normalize()).size()
full_daily_idx = pd.date_range(daily_orders.index.min(), daily_orders.index.max(), freq="D")
daily_orders = daily_orders.reindex(full_daily_idx, fill_value=0)
print(f"\nCalendar days spanned: {len(daily_orders)}")
print(f"Days with zero orders: {(daily_orders == 0).sum()} "
      f"({100 * (daily_orders == 0).mean():.1f}%)")
print(f"Mean orders/day: {daily_orders.mean():.2f} | CV: {daily_orders.std()/daily_orders.mean():.2f}")

fig, ax = plt.subplots()
daily_orders.rolling(7, min_periods=1).mean().plot(ax=ax, color="steelblue")
ax.set_title("Daily order *count* (7-day rolling mean) — volume over time")
ax.set_ylabel("Orders per day")
show_plot()

# %% [markdown]
# ### 3.1 Candidate granularities: daily vs weekly unit demand
#
# Daily retail series are often sparse and noisy for *aggregate* planning.
# We compare daily vs weekly **Quantity** sums with hard numbers, then pick
# one grain for the rest of the notebook.

# %%
demand_daily = (
    df.groupby(df["Order Date"].dt.normalize())["Quantity"]
    .sum()
    .rename("demand")
    .asfreq("D", fill_value=0)
    .sort_index()
)
demand_weekly = (
    df.set_index("Order Date")["Quantity"]
    .resample("W-SUN")
    .sum()
    .rename("demand")
    .sort_index()
)
# Drop leading/trailing partial zeros only at edges if empty — keep interior zeros
demand_weekly = demand_weekly.loc[demand_weekly.ne(0).idxmax() : demand_weekly.ne(0)[::-1].idxmax()]

def series_profile(s: pd.Series, name: str) -> dict:
    s = s.astype(float)
    prof = {
        "grain": name,
        "n": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "cv": float(s.std(ddof=1) / s.mean()) if s.mean() else np.nan,
        "pct_zero": float((s == 0).mean() * 100),
        "min": float(s.min()),
        "max": float(s.max()),
        "start": str(s.index.min().date()),
        "end": str(s.index.max().date()),
    }
    return prof

profiles = pd.DataFrame([
    series_profile(demand_daily, "daily"),
    series_profile(demand_weekly, "weekly"),
])
display(profiles)

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
demand_daily.plot(ax=axes[0], color="gray", alpha=0.7, title="Daily unit demand")
axes[0].set_ylabel("Units")
demand_weekly.plot(ax=axes[1], color="darkgreen", marker="o", ms=3, title="Weekly unit demand (W-SUN)")
axes[1].set_ylabel("Units")
show_plot()

# Decision rule: prefer weekly if daily has high zero share or much higher CV
daily_pct_zero = profiles.loc[profiles.grain == "daily", "pct_zero"].iloc[0]
daily_cv = profiles.loc[profiles.grain == "daily", "cv"].iloc[0]
weekly_cv = profiles.loc[profiles.grain == "weekly", "cv"].iloc[0]

if daily_pct_zero > 15 or daily_cv > weekly_cv * 1.15:
    GRAIN = "weekly"
    y_full = demand_weekly.copy()
    SEASONAL_PERIOD = 52  # annual seasonality on weekly data (if history allows)
    FREQ = "W-SUN"
    decision_reason = (
        f"Daily series has {daily_pct_zero:.1f}% zero-demand days and CV={daily_cv:.2f} "
        f"vs weekly CV={weekly_cv:.2f}. Weekly aggregation is more stable for inventory planning."
    )
else:
    GRAIN = "daily"
    y_full = demand_daily.copy()
    SEASONAL_PERIOD = 7
    FREQ = "D"
    decision_reason = (
        f"Daily series is dense enough (zero days={daily_pct_zero:.1f}%, CV={daily_cv:.2f}); "
        "keeping daily grain for finer-grained planning."
    )

print(f"\n>>> GRANULARITY DECISION: {GRAIN.upper()}")
print(decision_reason)
print(f"Series length: {len(y_full)} | {y_full.index.min().date()} → {y_full.index.max().date()}")

# %% [markdown]
# ### 3.2 Trend, seasonality, stationarity on the chosen series

# %%
# STL needs a consistent period that fits the sample length
stl_period = SEASONAL_PERIOD
if len(y_full) < 2 * stl_period + 1:
    stl_period = 7 if GRAIN == "weekly" else 7
    print(f"Series shorter than 2×{SEASONAL_PERIOD}; using STL period={stl_period}")

stl = STL(y_full.astype(float), period=stl_period, robust=True)
stl_res = stl.fit()

fig = stl_res.plot()
fig.set_size_inches(11, 7)
fig.suptitle(f"STL decomposition ({GRAIN}, period={stl_period})", y=1.01)
show_plot()

adf_stat, adf_p, *_ = adfuller(y_full.astype(float), autolag="AIC")
print(f"ADF statistic={adf_stat:.4f}, p-value={adf_p:.4g}")
if adf_p < 0.05:
    print("Reject H0 of unit root → series looks stationary (at 5%).")
else:
    print("Fail to reject unit root → trend/integration may be present; ARIMA-style differencing helps.")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
plot_acf(y_full.astype(float), lags=min(40, len(y_full) // 2 - 1), ax=axes[0])
plot_pacf(y_full.astype(float), lags=min(40, len(y_full) // 2 - 1), ax=axes[1], method="ywm")
axes[0].set_title("ACF — chosen demand series")
axes[1].set_title("PACF — chosen demand series")
show_plot()

# Light business breakdown (context only — forecast remains total units)
if "Category" in df.columns:
    by_cat = df.groupby("Category")["Quantity"].sum().sort_values(ascending=False)
    print("\nUnits by Category (full history):")
    display(by_cat.to_frame("units"))
if "Region" in df.columns:
    by_reg = df.groupby("Region")["Quantity"].sum().sort_values(ascending=False)
    print("Units by Region (full history):")
    display(by_reg.to_frame("units"))

# %% [markdown]
# ## 4. Target series + time-ordered holdout
#
# **No random shuffle.** The last `H` periods are the test set; everything
# before is train / TimesFM context. The same `H` is used for PyCaret's
# forecast horizon and TimesFM's `horizon`.

# %%
# Holdout: 8 weeks or ~28 days, but keep train reasonably long
if GRAIN == "weekly":
    H = 8
else:
    H = 28

if len(y_full) <= H + 20:
    H = max(4, len(y_full) // 5)
    print(f"Short series — reducing H to {H}")

y = y_full.astype(float).copy()
y.name = "demand"
y_train = y.iloc[:-H]
y_test = y.iloc[-H:]

print(f"Grain={GRAIN} | H={H} | seasonal_period candidate={SEASONAL_PERIOD}")
print(f"Train: {y_train.index.min().date()} → {y_train.index.max().date()} (n={len(y_train)})")
print(f"Test : {y_test.index.min().date()} → {y_test.index.max().date()} (n={len(y_test)})")
print(f"Train mean demand: {y_train.mean():.1f} units / period")
print(f"Test  mean demand: {y_test.mean():.1f} units / period")

fig, ax = plt.subplots()
y_train.plot(ax=ax, label="train", color="steelblue")
y_test.plot(ax=ax, label="test (holdout)", color="crimson")
ax.axvline(y_test.index[0], color="k", ls="--", lw=1, alpha=0.6)
ax.set_title(f"Unit demand series with time-ordered holdout (H={H}, {GRAIN})")
ax.set_ylabel("Units")
ax.legend()
show_plot()

# %%
def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train_arr: np.ndarray, m: int) -> dict:
    """MAE, RMSE, MAPE (epsilon-safe), MASE (seasonal naive scale)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train_arr = np.asarray(y_train_arr, dtype=float)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = float(np.mean(np.abs(err) / denom) * 100)
    m = max(1, min(m, len(y_train_arr) // 2))
    scale = np.mean(np.abs(y_train_arr[m:] - y_train_arr[:-m]))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = np.mean(np.abs(np.diff(y_train_arr))) + 1e-8
    mase = float(mae / scale)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "MASE": mase}

# %% [markdown]
# ## 5. Part 1 — PyCaret time-series survey (MASE-ranked)
#
# PyCaret 4.x uses an **experiment class**, not the old free functions
# (`setup` / `compare_models` as bare imports). Pattern:
#
# ```python
# exp = TimeSeriesExperiment(fh=H, session_id=42)
# exp.fit(y_train)                 # univariate Series
# result = exp.compare_models(sort="MASE")
# # result.best, result.leaderboard, result.ranked_ids
# ```
#
# We fit on **train only** so the holdout stays untouched. `sort="MASE"` is
# scale-independent and is PyCaret's recommended cross-series metric.

# %%
from pycaret.time_series import TimeSeriesExperiment

# Seasonal period for model selection:
# Full annual weekly seasonality (52) makes auto_arima/ETS extremely slow on ~200 points.
# Use a practical short cycle for the survey (monthly-ish weeks=4; daily=7).
if GRAIN == "weekly":
    sp_for_exp = 4
else:
    sp_for_exp = 7
if len(y_train) < 2 * sp_for_exp:
    sp_for_exp = max(1, len(y_train) // 4)
print(
    f"PyCaret seasonal_period={sp_for_exp} "
    f"(STL/ACF still used period≈{SEASONAL_PERIOD} for EDA; survey uses a faster cycle)"
)

exp = TimeSeriesExperiment(
    fh=H,
    session_id=SEED,
    fold=2,
    fold_strategy="expanding",
    seasonal_period=sp_for_exp,
    verbose=False,
    n_jobs=1,
)
exp.fit(y_train)

print("Registered turbo-eligible models (sample):")
try:
    model_table = exp.models()
    display(model_table)
except Exception as exc:  # noqa: BLE001
    print("models() unavailable:", exc)

# Focus the survey on classical families (fast + natively re-implementable).
# turbo=True alone still tries many ML-reduction models and can take a long time.
# auto_arima omitted from the survey for runtime; re-added only if we need a
# broader search. Classical set still covers ARIMA/ETS/Theta/naive baselines.
SURVEY_INCLUDE = [
    "naive",
    "snaive",
    "grand_means",
    "polytrend",
    "arima",
    "exp_smooth",
    "ets",
    "theta",
    "croston",
]
print(f"\nRunning compare_models(sort='MASE', include={SURVEY_INCLUDE})…")
compare_result = exp.compare_models(
    sort="MASE",
    include=SURVEY_INCLUDE,
    turbo=True,
    n_select=1,
    verbose=False,
)

leaderboard = compare_result.leaderboard.copy()
# Drop rows where MASE failed entirely
if "MASE" in leaderboard.columns:
    leaderboard_valid = leaderboard.dropna(subset=["MASE"]).reset_index(drop=True)
else:
    leaderboard_valid = leaderboard

print("\n=== PyCaret leaderboard (MASE-sorted; lower is better) ===")
display(leaderboard_valid)

if leaderboard_valid.empty or compare_result.best is None:
    raise RuntimeError(
        "PyCaret compare_models returned no successful models. "
        f"Raw leaderboard:\n{leaderboard}"
    )

winner_id = str(compare_result.ranked_ids[0])
winner_row = leaderboard_valid.iloc[0]
print(f"\n>>> PYCARET WINNER: {winner_id}")
print(winner_row.to_string())

# %% [markdown]
# ## 6. Production classical + foundation bake-off
#
# Research / production practice for retail unit demand:
#
# 1. **Annual seasonality** on weekly series (`m=52`) often dominates short-cycle models.
# 2. **Multiplicative Holt–Winters** handles year-end amplitude better than additive ETS with `m=4`.
# 3. **Validation-selected ensembles** and **rolling-origin** checks beat a single lucky cut.
# 4. **TimesFM zero-shot** remains a strong foundation baseline with quantile bands.
#
# PyCaret (above) is the educational survey. Below is the **production bake-off**
# implemented in `demand_forecast/` — native statsmodels/pmdarima + TimesFM 2.5,
# nested validation weights, holdout metrics, and rolling-origin robustness.

# %%
from demand_forecast import run_production_bakeoff
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats as scipy_stats

prod = run_production_bakeoff(
    y,
    grain=GRAIN,
    h=H,
    seed=SEED,
    n_rolling_origins=3,
    include_timesfm=True,
)

print("=== Production notes ===")
for note in prod.notes:
    print(" -", note)

print("\n=== Holdout leaderboard (sorted by MASE; lower is better) ===")
display(prod.leaderboard.round(4))

champ = prod.champion
print(f"\n>>> PRODUCTION CHAMPION: {champ.name} ({champ.kind})")
print(f"Details: {champ.details}")
print({k: round(v, 4) for k, v in champ.test_metrics.items()})
print(f"PI coverage on holdout: {champ.coverage}")

# Map series for plots
champ_fc = pd.Series(champ.point, index=y_test.index, name=champ.name)
champ_lo = pd.Series(champ.lower, index=y_test.index)
champ_hi = pd.Series(champ.upper, index=y_test.index)

# TimesFM row if present
tfm_row = next((c for c in prod.candidates if c.name.startswith("timesfm")), None)
native_like = next(
    (c for c in prod.candidates if c.kind == "classical" and c.name == champ.name),
    next((c for c in prod.candidates if c.kind == "classical"), None),
)

# Residual diagnostics on champion if classical residuals available
if champ.residuals is not None and len(champ.residuals.dropna()) > 10:
    resid = champ.residuals.dropna()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    resid.plot(ax=axes[0], title="Champion residuals")
    axes[0].axhline(0, color="k", lw=0.8)
    plot_acf(resid, lags=min(24, len(resid) // 2 - 1), ax=axes[1])
    plot_pacf(resid, lags=min(24, len(resid) // 2 - 1), ax=axes[2], method="ywm")
    axes[1].set_title("Residual ACF")
    axes[2].set_title("Residual PACF")
    show_plot()
    lb = acorr_ljungbox(resid, lags=[min(10, max(1, len(resid) // 5))], return_df=True)
    print("Ljung-Box (champion residuals):")
    display(lb)
else:
    print("Champion has no classical residual series (foundation/ensemble) — skip Ljung-Box.")

fig, ax = plt.subplots(figsize=(11, 4))
y_train.iloc[-min(60, len(y_train)):].plot(ax=ax, label="train (tail)", color="steelblue")
y_test.plot(ax=ax, label="actual test", color="black", lw=2)
champ_fc.plot(ax=ax, label=f"champion: {champ.name}", color="darkorange", lw=2)
ax.fill_between(y_test.index, champ_lo, champ_hi, color="darkorange", alpha=0.2, label="PI band")
if tfm_row is not None and tfm_row.name != champ.name:
    pd.Series(tfm_row.point, index=y_test.index).plot(
        ax=ax, label="TimesFM 2.5", color="purple", lw=1.5, alpha=0.85
    )
ax.set_title("Production champion forecast vs holdout")
ax.set_ylabel("Units")
ax.legend()
show_plot()

# %% [markdown]
# ## 7. Rolling-origin robustness (production check)
#
# A single 8-week cut can flatter a model. We re-fit candidates on multiple
# expanding origins and report mean MASE/MAE — closer to how ops would gate a model.

# %%
if prod.rolling is not None and len(prod.rolling):
    roll_mean = (
        prod.rolling.groupby("model")[["MAE", "RMSE", "MAPE", "MASE"]]
        .mean()
        .sort_values("MASE")
    )
    print("=== Rolling-origin mean metrics ===")
    display(roll_mean.round(4))
    fig, ax = plt.subplots(figsize=(10, 4))
    top = roll_mean.head(8)
    top["MASE"].plot(kind="barh", ax=ax, color="teal")
    ax.set_xlabel("Mean MASE across origins (lower better)")
    ax.set_title("Rolling-origin model robustness")
    show_plot()
else:
    print("No rolling-origin rows produced.")

# %% [markdown]
# ## 8. Side-by-side: PyCaret survey vs production bake-off vs TimesFM
#
# Educational PyCaret winner is still reported; **shipping decision** uses the
# production bake-off champion (validated multi-model search with proper seasonality).

# %%
# Pull key rows
rows = {}
rows[f"PyCaret survey winner ({winner_id}) — prior native path not re-used"] = {
    "note": "see production table for true classical performance"
}
# Use leaderboard excerpt
key_models = []
for c in prod.candidates:
    if c.name == champ.name or c.name.startswith("timesfm") or "seasonal_naive_m52" in c.name or "auto_arima" in c.name:
        key_models.append(c)
    if c.kind == "ensemble":
        key_models.append(c)

comp = prod.leaderboard.copy()
print("Full production leaderboard:")
display(comp.round(4))

print("\n=== How to read this ===")
print(
    f"Champion **{champ.name}** achieves holdout MASE={champ.test_metrics['MASE']:.4f}, "
    f"MAE={champ.test_metrics['MAE']:.2f}, MAPE={champ.test_metrics['MAPE']:.2f}%."
)
if tfm_row is not None:
    print(
        f"TimesFM zero-shot MASE={tfm_row.test_metrics['MASE']:.4f}. "
        "Foundation models shine on cold-start / many series; here a well-specified "
        "seasonal classical model can win on a single strong retail seasonal series."
    )
print(
    f"PyCaret survey (period={sp_for_exp}) selected **{winner_id}** under a short-cycle "
    "setting for speed — that is intentional education vs production bake-off with "
    "annual weekly seasonality candidates."
)

# Overlay top-3
fig, ax = plt.subplots(figsize=(11, 4.5))
y_train.iloc[-min(40, len(y_train)):].plot(ax=ax, color="steelblue", alpha=0.7, label="train tail")
y_test.plot(ax=ax, color="black", lw=2.5, label="actual")
for c in sorted(prod.candidates, key=lambda x: x.test_metrics["MASE"])[:3]:
    pd.Series(c.point, index=y_test.index).plot(ax=ax, lw=2, label=f"{c.name}")
ax.set_title("Top-3 production candidates vs actual holdout")
ax.set_ylabel("Units")
ax.legend()
show_plot()

# Persist metrics
results_dir = _ROOT / "data" / "results"
results_dir.mkdir(parents=True, exist_ok=True)
out_path = results_dir / "superstore_production_metrics.csv"
comp.to_csv(out_path)
print(f"Wrote {out_path}")

# %% [markdown]
# ## 9. Inventory-planning takeaway (production posture)

# %%
mean_actual = float(y_test.mean())
avg_point = float(champ_fc.mean())
avg_lo = float(champ_lo.mean())
avg_hi = float(champ_hi.mean())
# Service-oriented reorder signal: use upper band or point + residual sigma
service_units = float(np.maximum(champ_fc, champ_hi * 0.0 + champ_fc).mean())  # placeholder
# Prefer mean of upper PI as conservative weekly intake signal
conservative = float(champ_hi.mean())

print("=== Inventory-facing summary (production champion) ===")
print(f"Grain: {GRAIN} | H={H} | champion={champ.name}")
print(f"Holdout mean actual demand: {mean_actual:.1f} units / week")
print(f"Champion mean point forecast: {avg_point:.1f}")
print(f"Champion mean PI lower/upper: {avg_lo:.1f} / {avg_hi:.1f}")
print(f"MASE={champ.test_metrics['MASE']:.4f} | MAE={champ.test_metrics['MAE']:.2f} | bias={champ.test_metrics['bias']:.2f}")
print()
print(
    f"Ops read: plan baseline replenishment near **{avg_point:.0f} units/week**, "
    f"with a service-minded envelope up to ~**{avg_hi:.0f}** (upper PI). "
    f"Actual holdout averaged {mean_actual:.0f}, so a pure point policy would have "
    f"{'under' if avg_point < mean_actual else 'over'}-served this window by "
    f"{abs(mean_actual - avg_point):.0f} units/week on average. "
    f"Bias={champ.test_metrics['bias']:.1f} (negative ⇒ under-forecast). "
    "This is directional safety-buffer guidance, not a full (Q,R) policy."
)
print()
print(decision_reason)
print(f"PyCaret educational winner: {winner_id} | Production champion: {champ.name}")
print("Notebook 01 complete (production bake-off).")
