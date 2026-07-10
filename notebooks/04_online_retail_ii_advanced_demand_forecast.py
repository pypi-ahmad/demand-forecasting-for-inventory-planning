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
# # Advanced Demand Forecasting Tutorial — Online Retail II (v3)
#
# **Additive notebook.** Does **not** modify `02_online_retail_ii_*`.  
# Applies the same **v3 advanced stack** as notebook 03 on UCI Online Retail II.
#
# **v2 baseline to beat (same weekly H=8 holdout):**  
# HW multiplicative m=13 → MAE **~22.5k**, MAPE **~11.9%**, MASE **~0.56**.
#
# Extra Retail-specific choices:
# - Hierarchy: **Country** (top markets) bottom-up  
# - Holiday calendar: **UK (GB)**  
# - No Superstore-style Discount column — calendar Fourier + holidays only  

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

from demand_forecast.advanced import run_advanced_pipeline
from demand_forecast.advanced.evaluation import summarize_rolling
from demand_forecast.advanced.models_exog import forecast_hw_mul
from demand_forecast.metrics import forecast_metrics


def show_plot() -> None:
    plt.tight_layout()
    if IN_IPYTHON:
        plt.show()
    else:
        plt.show(block=False)
        plt.close("all")


# %% [markdown]
# ## 1. Load Online Retail II (same fallback path as notebook 02)

# %%
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
DATA_DIR = _ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
LOCAL_ZIP = DATA_DIR / "online_retail_ii.zip"

if LOCAL_ZIP.exists() and LOCAL_ZIP.stat().st_size > 1_000_000:
    zip_bytes = LOCAL_ZIP.read_bytes()
    print("Using cached zip", LOCAL_ZIP)
else:
    import requests

    print("Downloading UCI zip…")
    r = requests.get(UCI_ZIP_URL, timeout=300)
    r.raise_for_status()
    zip_bytes = r.content
    LOCAL_ZIP.write_bytes(zip_bytes)

with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
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
print("Clean shape:", df.shape)

y = (
    df.set_index("InvoiceDate")["Quantity"]
    .resample("W-SUN")
    .sum()
    .astype(float)
    .sort_index()
)
y = y.loc[y.ne(0).idxmax() : y.ne(0)[::-1].idxmax()]
print("Weekly series:", len(y), y.index.min().date(), "→", y.index.max().date())

# %% [markdown]
# ## 2. Advanced pipeline (hierarchy by Country)

# %%
result = run_advanced_pipeline(
    y,
    dataset="online_retail_ii",
    transactions=df,
    date_col="InvoiceDate",
    qty_col="Quantity",
    hierarchy_cols=["Country"],
    promo_series=None,
    country="GB",
    h=8,
    n_rolling=6,
    underage_cost=4.0,
    overage_cost=1.0,
    seed=SEED,
)

print("=== Notes ===")
for n in result.notes:
    print(" -", n)

print("\n=== Accuracy (MASE) ===")
display(result.leaderboard.round(4))
print("\n=== Inventory cost ===")
display(result.inventory_table.round(2))
print("Champion:", result.champion_name)

# %% [markdown]
# ## 3. v2 reference vs v3

# %%
y_train, y_test = result.y_train, result.y_test
H = result.h
# v2 used m=13 HW mul when train long enough
period = 13 if len(y_train) >= 2 * 13 + 2 else 8
v2_ref = forecast_hw_mul(y_train, H, period=period)
v2_metrics = forecast_metrics(
    y_test.values, v2_ref.point, y_train.values, mase_period=52 if len(y_train) > 60 else 4
)
adv = result.leaderboard.loc[result.champion_name]
compare = pd.DataFrame(
    {
        f"v2_hw_mul_m{period}": v2_metrics,
        f"v3_{result.champion_name}": {
            "MAE": float(adv["MAE"]),
            "RMSE": float(adv["RMSE"]),
            "MAPE": float(adv["MAPE"]),
            "sMAPE": float(adv.get("sMAPE", np.nan)),
            "MASE": float(adv["MASE"]),
            "bias": float(adv["bias"]),
        },
    }
).T
print("=== v2 vs v3 ===")
display(compare.round(4))

if "hierarchy_bottom_up" in result.forecasts:
    hm = forecast_metrics(
        y_test.values,
        result.forecasts["hierarchy_bottom_up"].point,
        y_train.values,
        mase_period=52 if len(y_train) > 60 else 4,
    )
    print("Hierarchy alone:", {k: round(v, 4) for k, v in hm.items()})

# %% [markdown]
# ## 4. Plots

# %%
idx = y_test.index
fig, ax = plt.subplots(figsize=(11, 4.5))
y_train.iloc[-40:].plot(ax=ax, label="train tail", color="steelblue", alpha=0.7)
y_test.plot(ax=ax, label="actual", color="black", lw=2.5)
pd.Series(v2_ref.point, index=idx).plot(ax=ax, label=f"v2 HW mul m={period}", color="gray", ls="--")
pd.Series(result.champion_point, index=idx).plot(
    ax=ax, label=f"v3 {result.champion_name}", color="darkorange", lw=2
)
ax.fill_between(idx, result.champion_lower, result.champion_upper, color="darkorange", alpha=0.2)
pd.Series(result.order_qty_sl90, index=idx).plot(ax=ax, label="SL0.9 order", color="green")
ax.set_title("Online Retail II advanced: actual vs v2 vs v3")
ax.set_ylabel("Units / week")
ax.legend()
show_plot()

# %% [markdown]
# ## 5. Rolling-origin summaries

# %%
for name, rdf in result.rolling_summary.items():
    print(f"\n--- {name} ---")
    display(summarize_rolling(rdf).round(3))

# %% [markdown]
# ## 6. Inventory takeaway + persist

# %%
print("Actual mean:", float(y_test.mean()))
print("Point mean:", float(np.mean(result.champion_point)))
print("SL0.9 order mean:", float(np.mean(result.order_qty_sl90)))
print(result.inventory_table.loc[result.champion_name])

out = _ROOT / "data" / "results"
out.mkdir(parents=True, exist_ok=True)
result.leaderboard.to_csv(out / "online_retail_ii_v3_accuracy.csv")
result.inventory_table.to_csv(out / "online_retail_ii_v3_inventory.csv")
compare.to_csv(out / "online_retail_ii_v2_vs_v3.csv")
print("Wrote", out)
print("Notebook 04 complete.")
