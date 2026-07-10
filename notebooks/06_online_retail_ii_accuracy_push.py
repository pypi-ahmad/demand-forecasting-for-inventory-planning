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
# # Accuracy Push (v4) — Online Retail II
#
# **Does not edit notebooks 01–04.** Goal: beat **v2 MAE ≈ 22,453** (HW mul m=13)
# on the same weekly H=8 holdout using hierarchy (Country), blends, calendar
# residual correction, and multi-window selection.
#
# ## Frozen baselines
#
# | Version | Model | MAE | MAPE | MASE |
# |---------|-------|-----:|-----:|-----:|
# | v1 | ARIMA m=4 | 69,829 | 38.5% | 1.83 |
# | v1 | TimesFM | 31,732 | 16.9% | 0.83 |
# | v2/v3 | HW mul m=13 | **22,453** | **11.9%** | **0.56** |

# %%
from __future__ import annotations

import io
import sys
import warnings
import zipfile
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
# ## 1. Load data (same as notebook 02)

# %%
UCI_ZIP = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
DATA = _ROOT / "data"
DATA.mkdir(exist_ok=True)
LOCAL = DATA / "online_retail_ii.zip"
if LOCAL.exists() and LOCAL.stat().st_size > 1_000_000:
    raw = LOCAL.read_bytes()
else:
    import requests

    raw = requests.get(UCI_ZIP, timeout=300).content
    LOCAL.write_bytes(raw)

with zipfile.ZipFile(io.BytesIO(raw)) as zf:
    with zf.open("online_retail_II.xlsx") as fh:
        xl = pd.ExcelFile(fh)
        df = pd.concat([xl.parse(s) for s in xl.sheet_names], ignore_index=True)

df["Invoice"] = df["Invoice"].astype(str)
df = df[~df["Invoice"].str.startswith("C")]
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df = df.dropna(subset=["InvoiceDate", "Quantity", "Price"])
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()

y = (
    df.set_index("InvoiceDate")["Quantity"]
    .resample("W-SUN")
    .sum()
    .astype(float)
    .sort_index()
)
y = y.loc[y.ne(0).idxmax() : y.ne(0)[::-1].idxmax()]
print(len(y), y.index.min().date(), "→", y.index.max().date())

# %% [markdown]
# ## 2. Accuracy-push pipeline

# %%
result = run_accuracy_push(
    y,
    df,
    dataset="online_retail_ii",
    date_col="InvoiceDate",
    qty_col="Quantity",
    hierarchy_specs=[["Country"]],
    promo_col=None,
    country="GB",
    h=8,
    n_origins=6,
    seed=SEED,
)

print("=== Notes ===")
for n in result.notes:
    print(" -", n)

print("\n=== Multi-window mean scores ===")
display(result.multiwindow_scores.head(12).round(4))

print("\n=== Holdout leaderboard ===")
display(result.leaderboard.round(4))

print("MW champion:", result.champion_name)
print("Holdout best:", result.leaderboard.index[0])

# %% [markdown]
# ## 3. v1 / v2 / v4 comparison

# %%
H = result.h
y_train, y_test = result.y_train, result.y_test
# v2 reference m=13
m = ExponentialSmoothing(
    y_train.clip(lower=1e-3),
    trend="add",
    seasonal="mul",
    seasonal_periods=13,
    initialization_method="estimated",
).fit(optimized=True)
v2 = np.clip(np.asarray(m.forecast(H)), 0, None)
v2_m = forecast_metrics(y_test.values, v2, y_train.values, mase_period=52)

v4_name = result.champion_name
v4_m = forecast_metrics(
    y_test.values, result.champion_point, y_train.values, mase_period=52
)
hb = str(result.leaderboard.index[0])
hb_m = forecast_metrics(
    y_test.values, result.forecasts[hb], y_train.values, mase_period=52
)

compare = pd.DataFrame(
    {
        "v2_hw_mul_m13": v2_m,
        f"v4_mw_{v4_name}": v4_m,
        f"v4_holdout_{hb}": hb_m,
    }
).T
display(compare.round(4))
for label, md in [(f"v4 MW {v4_name}", v4_m), (f"v4 holdout {hb}", hb_m)]:
    print(
        f"{label} vs v2: ΔMAE={(md['MAE']-v2_m['MAE'])/v2_m['MAE']*100:+.1f}% "
        f"ΔMAPE={(md['MAPE']-v2_m['MAPE'])/v2_m['MAPE']*100:+.1f}%"
    )

# %% [markdown]
# ## 4. Plot + persist

# %%
idx = y_test.index
fig, ax = plt.subplots(figsize=(11, 4.5))
y_train.iloc[-40:].plot(ax=ax, label="train tail", color="steelblue", alpha=0.7)
y_test.plot(ax=ax, label="actual", color="black", lw=2.5)
pd.Series(v2, index=idx).plot(ax=ax, label="v2 HW mul m=13", color="gray", ls="--")
pd.Series(result.champion_point, index=idx).plot(
    ax=ax, label=f"v4 MW: {v4_name}", color="darkorange", lw=2
)
if hb != v4_name:
    pd.Series(result.forecasts[hb], index=idx).plot(
        ax=ax, label=f"v4 holdout: {hb}", color="green", lw=1.8
    )
ax.set_title("Online Retail II accuracy push")
ax.set_ylabel("Units / week")
ax.legend()
show_plot()

out = _ROOT / "data" / "results"
out.mkdir(parents=True, exist_ok=True)
result.leaderboard.to_csv(out / "online_retail_ii_v4_holdout.csv")
result.multiwindow_scores.to_csv(out / "online_retail_ii_v4_multiwindow.csv")
compare.to_csv(out / "online_retail_ii_v2_v4_compare.csv")
print("Wrote", out)
print("Notebook 06 complete.")
