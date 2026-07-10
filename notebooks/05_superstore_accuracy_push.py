# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: demand-forecast-project
#     language: python
#     name: demand-forecast-project
# ---

# %% [markdown]
# # Accuracy Push (v4) — Superstore
#
# **Does not edit notebooks 01–04.** Goal: beat **v2 MAE ≈ 48.12** on the same
# weekly Superstore holdout (H=8) with stronger **signal** (hierarchy) and
# **multi-window selection** (not one lucky cut).
#
# ## Why these techniques
#
# | Technique | Why it can beat pure total-level HW |
# |-----------|-------------------------------------|
# | **Category × Region bottom-up** | Different categories/regions have different seasonal shapes; sum recovers total with richer structure |
# | **Explosion caps** | Multiplicative HW on sparse bottoms can explode; cap at 3× historical max |
# | **Volume scaling / blends** | Reconcile hierarchy to univariate total level when needed |
# | **Calendar residual correction** | Holidays/promo explain leftover after HW |
# | **Multi-window mean MAE** | Champion chosen by average error across rolling origins, then scored on final holdout |
#
# ## Frozen baselines (same holdout)
#
# | Version | Model | MAE | MAPE | MASE |
# |---------|-------|-----:|-----:|-----:|
# | v1 | ETS short-cycle | 104.8 | 26.6% | 1.32 |
# | v1 | TimesFM | 78.8 | 18.4% | ~0.99 |
# | v2/v3 | HW mul m=52 | **48.12** | **11.44%** | **0.80** |

# %%
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch

try:
    from IPython import get_ipython
    from IPython.display import display

    IN_IPYTHON = get_ipython() is not None
except ImportError:
    IN_IPYTHON = False

    def display(obj):
        print(obj)

if not IN_IPYTHON:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (11, 4)

_ROOT = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

from demand_forecast.accuracy_push import run_accuracy_push
from demand_forecast.metrics import forecast_metrics
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def show_plot() -> None:
    plt.tight_layout()
    if IN_IPYTHON:
        plt.show()
    else:
        plt.show(block=False)
        plt.close("all")


# %% [markdown]
# ## 1. Data (identical Superstore construction as 01/03)

# %%
URL = (
    "https://raw.githubusercontent.com/yajasarora/"
    "Superstore-Sales-Analysis-with-Tableau/master/"
    "Superstore%20sales%20dataset.csv"
)
df = pd.read_csv(URL, encoding="utf-8-sig")
df.columns = [c.strip() for c in df.columns]
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce", dayfirst=True)
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Discount"] = pd.to_numeric(df.get("Discount", 0), errors="coerce").fillna(0.0)
df = df.dropna(subset=["Order Date", "Quantity"])
df = df[df["Quantity"] > 0].copy()

y = (
    df.set_index("Order Date")["Quantity"]
    .resample("W-SUN")
    .sum()
    .astype(float)
    .sort_index()
)
y = y.loc[y.ne(0).idxmax() : y.ne(0)[::-1].idxmax()]
print(y.shape, y.index.min().date(), "→", y.index.max().date())

# %% [markdown]
# ## 2. Run accuracy-push pipeline

# %%
result = run_accuracy_push(
    y,
    df,
    dataset="superstore",
    date_col="Order Date",
    qty_col="Quantity",
    hierarchy_specs=[["Category"], ["Region"], ["Category", "Region"]],
    promo_col="Discount",
    country="US",
    h=8,
    n_origins=6,
    seed=SEED,
)

print("=== Notes ===")
for n in result.notes:
    print(" -", n)

print("\n=== Multi-window mean scores (champion selection) ===")
display(result.multiwindow_scores.head(12).round(4))

print("\n=== Final holdout leaderboard (sorted by MAE) ===")
display(result.leaderboard.round(4))

print("\n>>> v4 multi-window champion:", result.champion_name)
if len(result.leaderboard):
    print(">>> holdout best MAE model:", result.leaderboard.index[0])

# %% [markdown]
# ## 3. Compare v1 / v2 / v4 on the **same** holdout

# %%
H = result.h
y_train, y_test = result.y_train, result.y_test

# v2 reference
m = ExponentialSmoothing(
    y_train.clip(lower=1e-3),
    trend="add",
    seasonal="mul",
    seasonal_periods=52,
    initialization_method="estimated",
).fit(optimized=True)
v2 = np.clip(np.asarray(m.forecast(H)), 0, None)
v2_m = forecast_metrics(y_test.values, v2, y_train.values, mase_period=52)

v4_name = result.champion_name
v4_point = result.champion_point
v4_m = forecast_metrics(y_test.values, v4_point, y_train.values, mase_period=52)

# holdout-best if different
holdout_best = str(result.leaderboard.index[0])
hb_point = result.forecasts[holdout_best]
hb_m = forecast_metrics(y_test.values, hb_point, y_train.values, mase_period=52)

# v1-ish short cycle HW m4
m4 = ExponentialSmoothing(
    y_train.clip(lower=1e-3),
    trend="add",
    seasonal="mul",
    seasonal_periods=4,
    initialization_method="estimated",
).fit(optimized=True)
v1_like = np.clip(np.asarray(m4.forecast(H)), 0, None)
v1_m = forecast_metrics(y_test.values, v1_like, y_train.values, mase_period=52)

compare = pd.DataFrame(
    {
        "v1_like_hw_mul_m4": v1_m,
        "v2_hw_mul_m52": v2_m,
        f"v4_mw_champion_{v4_name}": v4_m,
        f"v4_holdout_best_{holdout_best}": hb_m,
    }
).T
print("=== v1 / v2 / v4 comparison (holdout) ===")
display(compare.round(4))

# improvement vs v2
for label, mdict in [
    (f"v4 multi-window ({v4_name})", v4_m),
    (f"v4 holdout-best ({holdout_best})", hb_m),
]:
    d_mae = (mdict["MAE"] - v2_m["MAE"]) / v2_m["MAE"] * 100
    d_mape = (mdict["MAPE"] - v2_m["MAPE"]) / v2_m["MAPE"] * 100
    print(f"{label} vs v2: ΔMAE={d_mae:+.1f}%  ΔMAPE={d_mape:+.1f}%")

# %% [markdown]
# ## 4. Plot

# %%
idx = y_test.index
fig, ax = plt.subplots(figsize=(11, 4.5))
y_train.iloc[-40:].plot(ax=ax, label="train tail", color="steelblue", alpha=0.7)
y_test.plot(ax=ax, label="actual", color="black", lw=2.5)
pd.Series(v2, index=idx).plot(ax=ax, label="v2 HW mul m=52", color="gray", ls="--")
pd.Series(v4_point, index=idx).plot(ax=ax, label=f"v4 MW: {v4_name}", color="darkorange", lw=2)
if holdout_best != v4_name:
    pd.Series(hb_point, index=idx).plot(
        ax=ax, label=f"v4 holdout-best: {holdout_best}", color="green", lw=1.8
    )
ax.fill_between(idx, result.champion_lower, result.champion_upper, color="darkorange", alpha=0.15)
ax.set_title("Superstore accuracy push: actual vs v2 vs v4")
ax.set_ylabel("Units / week")
ax.legend()
show_plot()

# %% [markdown]
# ## 5. Persist + takeaways

# %%
out = _ROOT / "data" / "results"
out.mkdir(parents=True, exist_ok=True)
result.leaderboard.to_csv(out / "superstore_v4_holdout.csv")
result.multiwindow_scores.to_csv(out / "superstore_v4_multiwindow.csv")
compare.to_csv(out / "superstore_v1_v2_v4_compare.csv")
print("Wrote", out)

print(
    """
Takeaways
- v4 keeps v2 HW mul m=52 in the candidate set.
- Category×Region bottom-up (and scaled/blended variants) often improves peak weeks
  because bottoms carry different seasonal shapes.
- Multi-window selection prefers models that work on average, not only the last cut.
- If multi-window champion ≠ holdout-best, report both (ops vs leaderboard honesty).
"""
)
print("Notebook 05 complete.")
